import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNConv(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_rels,
                 num_bases=-1,
                 bias=False,
                 activation=None,
                 dropout=0.5,
                 is_output_layer=False):
        r"""The relational graph convolutional operator from the `"Modeling
        Relational Data with Graph Convolutional Networks"
        <https://arxiv.org/abs/1703.06103>`_ paper

        Propagation model:
        (1) $h_i^{l+1} = \sigma\left(\sum_{r\in R}\sum_{j\in N^r_i}\frac{1}{c_{i,r}}W_r^{(l)}h_j^{(l)}+
        \underbrace{W_0^{(l)}h_i^{(l)}}_{\text{self-connection}}\right)$

        where
        - $N^r_i$ denotes the set of neighbor indices of node $i$ under relation $r \in R$,
        - $c_{i,r}$ is a problem-specific normalization constant that can either be learned or chosen in advance
          (such as $c_{i,r} = |N_i^r|$).

        Neural network layer update: evaluate message passing update in parallel for every node $i \in V$.

        Parameter sharing for highly- multi-relational data: basis decomposition of relation-specific weight matrices
        (2) $W_r^{(l)} = \sum^B_{b=1}a^{(l)}_{r,b}V_b^{(l)}$

        Linear combination of basis transformations $V_b^{(l)} \in \mathbb{R}^{d^{(l+1)}\times d^{(l)}}$ with learnable
        coefficients $a^{(l)}_{r,b}$ such that only the coefficients depend on $r$. $B$, the number of basis functions,
        is a hyperparameter.

        :param input_dim: Input dimension
        :param output_dim: Output dimension
        :param num_rels: Number of relation types
        :param num_bases: Number of bases used in basis decomposition of relation-specific weight matrices
        :param bias: Optional additive bias
        :param activation: Activation function
        :param dropout: Dropout
        :param is_output_layer: Indicates whether this layer is the output layer
        """
        super(RGCNConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.dropout = dropout
        self.is_output_layer = is_output_layer

        # Number of bases for the basis decomposition can be less or equal to the number of relation types
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # Weight bases in equation (2), V_b if self.num_bases < self.num_rels, W_r if self.num_bases == self.num_rels
        self.weight = nn.Parameter(torch.Tensor(self.num_bases * self.input_dim, self.output_dim))

        # Use basis decomposition otherwise if num_bases = num_rels we can just use one weight matrix per relation type
        if self.num_bases < self.num_rels:
            # linear combination coefficients a^{(l)}_{r, b} in equation (2)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        if self.bias:
            self.b = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize trainable parameters, see following link for explanation:
        # https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
        # Xavier initialization: improved weight initialization method enabling quicker convergence and higher accuracy
        # gain is an optional scaling factor, here we use the recommended gain value for the given nonlinearity function
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.b, gain=nn.init.calculate_gain('relu'))

    def forward(self, x, adj_t):
        supports = []   # supports holds hidden states of the nodes
        num_nodes = adj_t[0].shape[0]
        for i, adj in enumerate(adj_t):
            # Matrix multiplication of relation type specific adjacency matrices
            # with x, the node embeddings or hidden states of the nodes from the previous layer
            if x is not None:
                supports.append(torch.spmm(adj, x))
            else:
                supports.append(adj)
        # list of (num_nodes, num_nodes) tensors -> (num_nodes, num_nodes*num_rel)
        supports = torch.cat(supports, dim=1)

        # Calculate relation specific weight matrices
        if self.num_bases < self.num_rels:
            # Generate all weights from bases as in equation (2)
            weight = self.weight.reshape(self.num_bases, self.input_dim, self.output_dim).permute(1, 0, 2)

            # Matrix product: learnable coefficients a_{r, b} and basis transformations V_b
            # (self.num_rels, self.num_bases) x (self.input_dim, self.num_bases, self.output_dim)
            weight = torch.matmul(self.w_comp, weight)  # (self.input_dim, self.num_rels, self.output_dim)
            weight = weight.reshape(self.input_dim * self.num_rels, self.output_dim)
        else:
            weight = self.weight

        out = torch.spmm(supports, weight)  # (num_nodes, num_rels)

        # If x is None add dropout to output, by elementwise multiplying with column vector of ones,
        # with dropout applied to the vector of ones.
        if x is None:
            temp = torch.ones(num_nodes).to(out.device)
            temp_drop = F.dropout(temp, self.dropout)
            out = (out.transpose(1, 0) * temp_drop).transpose(1, 0)

        if self.bias:
            out += self.b

        out = self.activation(out)
        return out
