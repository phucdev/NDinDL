import argparse
import os.path as osp
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from RGCNConv import RGCNConv
import utils


# Somewhat inspired by: https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/4_rgcn.html
class RGCN(torch.nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels,
                 num_bases=-1, num_hidden_layers=1, dropout=0.5):
        """
        Implementation of R-GCN from the `"Modeling
        Relational Data with Graph Convolutional Networks"
        <https://arxiv.org/abs/1703.06103>`_ paper

        :param num_nodes: Number of nodes (input dimension)
        :param h_dim: Hidden dimension
        :param out_dim: Output dimension
        :param num_rels: Number of relation types
        :param num_bases: Number of basis functions
        :param num_hidden_layers: Number of hidden layers
        :param dropout: Dropout probability
        """
        super(RGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()
        # create rgcn layers
        self.build_model()

    def build_model(self):
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    def build_input_layer(self):
        return RGCNConv(self.num_nodes, self.h_dim, self.num_rels, self.num_bases, activation=F.relu)

    def build_hidden_layer(self):
        return RGCNConv(self.h_dim, self.h_dim, self.num_rels, self.num_bases, activation=F.relu)

    def build_output_layer(self):
        return RGCNConv(self.h_dim, self.out_dim, self.num_rels, self.num_bases, activation=partial(F.softmax, dim=1))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj_t):
        out = x
        for layer in self.layers:
            out = layer(out, adj_t)
            if not layer.is_output_layer:
                out = F.dropout(out, self.dropout, self.training)
        return out


def train(model, x, adj_t, optimizer, loss_fn, train_idx, train_y):
    model.train()

    # Zero grad the optimizer
    optimizer.zero_grad()
    # Feed the data into the model
    out = model(x, adj_t)
    # Feed the sliced output and label to loss_fn
    labels = torch.LongTensor(train_y).to(out.device)
    loss = loss_fn(out[train_idx], labels)

    # Backpropagation, optimizer
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, x, adj_t, train_idx, train_y, test_idx, test_y):
    model.eval()

    # Output of model on all data
    out = model(x, adj_t)
    # Get predicted class labels
    pred = out.argmax(dim=-1).cpu()

    # Evaluate prediction accuracy
    train_acc = pred[train_idx].eq(train_y).to(torch.float).mean()
    test_acc = pred[test_idx].eq(test_y).to(torch.float).mean()
    return train_acc.item(), test_acc.item()


def main(args):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args["dataset"].lower() + '.pickle')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    A = data['A']
    y = data['y']
    train_idx = data['train_idx']
    test_idx = data['test_idx']

    num_nodes = y.shape[0]
    num_classes = y.shape[1]
    num_rels = len(A)

    y = np.squeeze(np.asarray(y.argmax(axis=-1))).astype('int64')
    train_y = torch.tensor(y[train_idx])
    test_y = torch.tensor(y[test_idx])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RGCN(
        num_nodes=num_nodes,
        h_dim=args["h_dim"],
        out_dim=num_classes,
        num_rels=num_rels,
        num_bases=args["num_bases"],
        dropout=args["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["l2"])
    loss_fn = nn.CrossEntropyLoss()

    adj_t = []
    for a in A:
        nor_a = utils.normalize(a)
        # if len(nor_a.nonzero()[0]) > 0:
        #     tensor_a = to_sparse_tensor(nor_a)
        #     adj_t.append(tensor_a.to(device))
        tensor_a = utils.to_sparse_tensor(nor_a)
        adj_t.append(tensor_a.to(device))
    x = None

    for epoch in range(1, args["epochs"] + 1):
        loss = train(model, x, adj_t, optimizer, loss_fn, train_idx, train_y)
        train_acc, test_acc = test(model, x, adj_t, train_idx, train_y, test_idx, test_y)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} '
              f'Test: {test_acc:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['AIFB', 'MUTAG', 'BGS', 'AM'], default='AIFB',
                        help="Dataset string ('AIFB', 'MUTAG', 'BGS', 'AM')")
    parser.add_argument('--h_dim', type=int, default=16, help='Number of hidden units')
    parser.add_argument('--num_bases', type=int, default=-1, help='Number of bases used (-1: all')
    parser.add_argument('--num_hidden_layers', type=int, default=0, help='Number of hidden layers')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate')
    parser.add_argument('--lr', type=int, default=0.01, help='Learning rate')
    parser.add_argument('--l2', type=int, default=0., help='Weight decay')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    arguments = parser.parse_args()

    args_dict = {
        'dataset': arguments.dataset,
        'h_dim': arguments.h_dim,
        'num_bases': arguments.num_bases,
        'num_hidden_layers': arguments.num_hidden_layers,
        'dropout': arguments.dropout,
        'lr': arguments.lr,
        'l2': arguments.l2,
        'epochs': arguments.epochs,
    }
    print(f"Model config:\n{args_dict}")
    main(args_dict)
