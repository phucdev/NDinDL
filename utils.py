import numpy as np
import scipy.sparse as sp


# https://github.com/mjDelta/relation-gcn-pytorch/blob/master/utils.py
def normalize(matrix):
    row_sum = np.array(matrix.sum(axis=1)).flatten()
    row_sum_inv = 1. / row_sum
    row_sum_inv[np.isinf(row_sum_inv)] = 0
    D_inv = sp.diags(row_sum_inv)
    out = D_inv.dot(matrix).tocsr()
    return out


def get_adj_t(data):
    adj_t = None

    num_rels = data.num_rels
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    edge_type = data.edge_type

    # TODO Build adjacency matrix for every relation type
    #  https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy-sparse-csr-matrix
    #  1. Build #num_rels adjacency matrices of dimension num_nodes x num_nodes filled with zeros
    #  2. Go through edge_index and edge_type and insert ones into adjacency matrices
    #  Notes: Look at the link for tips on how to construct the csr_matrix

    return adj_t
