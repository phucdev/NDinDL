import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# https://github.com/masakicktashiro/rgcn_pytorch_implementation/blob/master/layers.py
class RGCNConv(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 support=1,
                 featureless=True,
                 weights=None, num_bases=-1,
                 bias=False, dropout=0.):
        super(RGCNConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim  # number of features per node
        self.support = support  # filter support / number of weights
        self.featureless = featureless  # use/ignore input features
        self.dropout = dropout
        self.w_regularizer = nn.L1Loss()

        assert support >= 1

        self.bias = bias
        self.initial_weights = weights
        self.num_bases = num_bases

        # these will be defined during build()
        if self.num_bases > 0:
            self.W = nn.Parameter(
                torch.empty(self.input_dim * self.num_bases, self.output_dim, dtype=torch.float32, device=device)
            )
            self.W_comp = nn.Parameter(
                torch.empty(self.support, self.num_bases, dtype=torch.float32, device=device)
            )
            nn.init.xavier_uniform_(self.W_comp)
        else:
            self.W = nn.Parameter(
                torch.empty(self.input_dim * self.support, self.output_dim, dtype=torch.float32, device=device)
            )
        nn.init.xavier_uniform_(self.W)

        if self.bias:
            self.b = nn.Parameter(torch.empty(self.output_dim, dtype=torch.float32, device=device))
            nn.init.xavier_uniform_(self.b)

        self.dropout = nn.Dropout(dropout)

    def get_output_shape_for(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.output_dim)
        return output_shape  # (batch_size, output_dim)

    def forward(self, inputs):
        features = torch.tensor(inputs[0], dtype=torch.float32, device=device)
        A = inputs[1:]  # list of basis functions
        A = [torch.sparse_coo_tensor(torch.LongTensor(a.nonzero()),
                                     torch.FloatTensor(sparse.find(a)[-1]),
                                     torch.Size(a.shape)).to(device)
             if len(sparse.find(a)[-1]) > 0 else torch.sparse_coo_tensor(a.shape[0], a.shape[1])
             for a in A]
        # convolve
        if not self.featureless:
            supports = list()
            for i in range(self.support):
                supports.append(torch.spmm(A[i], features))
            supports = torch.cat(supports, dim=1)
        else:
            values = torch.cat([i._values() for i in A], dim=-1)
            indices = torch.cat([torch.cat([j._indices()[0].reshape(1, -1),
                                            (j._indices()[1] + (i * self.input_dim)).reshape(1, -1)])
                                 for i, j in enumerate(A)], dim=-1)
            supports = torch.sparse_coo_tensor(indices, values, torch.Size([A[0].shape[0],
                                                                            len(A) * self.input_dim]))
        num_nodes = supports.shape[0]
        if self.num_bases > 0:
            # self.W = self.W.reshape(
            #                   (self.num_bases, self.input_dim, self.output_dim))
            # self.W = self.W.permute((1, 0, 2)) # (self.input_dim, self.num_bases, self.output_dim)
            V = torch.matmul(self.W_comp,
                             self.W.reshape(self.num_bases, self.input_dim, self.output_dim).permute(1, 0, 2))
            V = torch.reshape(V, (self.support * self.input_dim, self.output_dim))
            output = torch.spmm(supports, V)
        else:
            output = torch.spmm(supports, self.W)

        # if featureless add dropout to output, by elementwise matmultiplying with column vector of ones,
        # with dropout applied to the vector of ones.
        if self.featureless:
            tmp = torch.ones(num_nodes)
            tmp_do = self.dropout(tmp)
            output = (output.transpose(1, 0) * tmp_do).transpose(1, 0)

        if self.bias:
            output += self.b
        return F.relu(output)
