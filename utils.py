import numpy as np
import scipy.sparse as sp
import torch


# https://github.com/mjDelta/relation-gcn-pytorch/blob/master/utils.py
def normalize(adj_matrix):
    """
    Normalizes the adjacency matrix
    :param adj_matrix: Adjacency matrix
    :return:
        out: Normalized adjacency matrix
    """
    node_degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    # Essentially 1. / node_degrees, while avoiding division by zero warning
    norm_const = np.divide(np.ones_like(node_degrees), node_degrees, out=np.zeros_like(node_degrees),
                           where=node_degrees != 0)
    D_inv = sp.diags(norm_const)
    out = D_inv.dot(adj_matrix).tocsr()
    return out


# https://github.com/mjDelta/relation-gcn-pytorch/blob/master/train.py
def to_sparse_tensor(sparse_array):
    """
    Converts sparse array (normalized adjacency matrix) to sparse tensor
    :param sparse_array: Sparse array (normalized adjacency matrix)
    :return:
        sparse_tensor: Converted sparse tensor
    """
    if len(sp.find(sparse_array)[-1]) > 0:
        # Get indices and values of nonzero elements in matrix
        v = torch.FloatTensor(sp.find(sparse_array)[-1])
        i = torch.LongTensor(sparse_array.nonzero())
        shape = sparse_array.shape
        sparse_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    else:
        sparse_tensor = torch.sparse_coo_tensor(sparse_array.shape[0], sparse_array.shape[1])
    return sparse_tensor


def get_adjacency_matrices(data):
    """
    Converts torch_geometric.datasets.entities data to relation type specific adjacency matrices
    :param data: torch_geometric.datasets.entities data
    :return:
        A: list of relation type specific adjacency matrices
    """
    num_rels = data.num_rels
    num_nodes = data.num_nodes

    A = []
    source_nodes = data.edge_index[0].numpy()
    target_nodes = data.edge_index[1].numpy()

    # Get edges for given (relation) edge type and construct adjacency matrix
    for i in range(num_rels):
        indices = np.argwhere(np.asarray(data.edge_type) == i).squeeze(axis=1)
        r_source_nodes = source_nodes[indices]
        r_target_nodes = target_nodes[indices]
        a = sp.csr_matrix((np.ones(len(indices)), (r_source_nodes, r_target_nodes)), shape=(num_nodes, num_nodes))
        A.append(a)

    return A
