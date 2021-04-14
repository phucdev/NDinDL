import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
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
    print("Converting torch_geometric.datasets.entities data to relation type specific adjacency matrices")
    num_rels = data.num_rels
    num_nodes = data.num_nodes

    A = [np.zeros((num_nodes, num_nodes)) for _ in range(num_rels)]

    edges = list(zip(data.edge_index[0].numpy(), data.edge_index[1].numpy()))
    for rel, (src, dst) in zip(data.edge_type, edges):
        A[rel][src][dst] = 1

    for i, m in tqdm(enumerate(A)):
        A[i] = sp.csr_matrix(m)

    return A
