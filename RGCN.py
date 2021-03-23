import argparse
import os.path as osp
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.utils.num_nodes import maybe_num_nodes
from RGCNConv import RGCNConv


# Somewhat inspired by: https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/4_rgcn.html
class RGCN(torch.nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels,
                 num_bases=-1, num_hidden_layers=1):
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
        """
        super(RGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        self.layers = nn.ModuleList()
        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

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

    # initialize feature for each node
    def create_features(self):
        features = torch.arange(self.num_nodes)
        return features

    def build_input_layer(self):
        return RGCNConv(self.num_nodes, self.h_dim, self.num_rels, self.num_bases,
                        activation=F.relu, is_input_layer=True)

    def build_hidden_layer(self):
        return RGCNConv(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                        activation=F.relu)

    def build_output_layer(self):
        return RGCNConv(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
                        activation=partial(F.softmax, dim=1))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj_t):
        out = None
        # TODO
        #  Note:
        #  1. Construct the network as described in the paper/ tutorial
        #  2. torch.nn.functional.relu and torch.nn.functional.dropout are useful
        #  More information please refer to the documentation:
        #  https://pytorch.org/docs/stable/nn.functional.html
        #  3. Don't forget to set F.dropout training to self.training
        return out


def train(model, data, optimizer, loss_fn):
    model.train()

    # Zero grad the optimizer
    optimizer.zero_grad()
    # Feed the data into the model
    out = model(data.edge_index, data.edge_type)
    # Feed the sliced output and label to loss_fn
    loss = loss_fn(out[data.train_idx], data.train_y)

    # Backpropagation, optimizer
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()

    # Output of model on all data
    out = model(data.edge_index, data.edge_type)
    # Get predicted class labels
    pred = out.argmax(dim=-1)

    # Evaluate prediction accuracy
    train_acc = pred[data.train_idx].eq(data.train_y).to(torch.float).mean()
    test_acc = pred[data.test_idx].eq(data.test_y).to(torch.float).mean()
    return train_acc.item(), test_acc.item()


def main(args):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities')
    dataset = Entities(path, args["dataset"])
    data = dataset[0]

    data.num_nodes = maybe_num_nodes(data.edge_index)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RGCN(
        num_nodes=data.num_nodes,
        h_dim=args["h_dim"],
        out_dim=dataset.num_classes,
        num_rels=dataset.num_relations,
        num_bases=args["num_bases"]
    ).to(device)
    data = data.to(device)

    # TODO: preprocess data
    #  Data(edge_index=[2, 58086], edge_type=[58086], test_idx=[36], test_y=[36], train_idx=[140], train_y=[140])
    #  edge_index: edge given by source and target node

    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=0.0005)
    loss_fn = F.nll_loss

    for epoch in range(1, args["epochs"] + 1):
        loss = train(model, data, optimizer, loss_fn)
        train_acc, test_acc = test()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} '
              f'Test: {test_acc:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['AIFB', 'MUTAG'])
    parser.add_argument('--h_dim', type=int, default=16)
    parser.add_argument('--num_bases', type=int, default=10)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=int, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    arguments = parser.parse_args()

    args_dict = {
        'dataset': arguments.dataset,
        'h_dim': arguments.h_dim,
        'num_bases': arguments.num_bases,
        'num_hidden_layers': arguments.num_hidden_layers,
        'dropout': 0.5,
        'lr': 0.01,
        'epochs': arguments.epochs,
    }
    main(args_dict)
