import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import torch


# https://github.com/mjDelta/relation-gcn-pytorch/blob/master/utils.py
def normalize(matrix):
    row_sum = np.array(matrix.sum(axis=1)).flatten()
    row_sum_inv = np.divide(np.ones_like(row_sum), row_sum, out=np.zeros_like(row_sum), where=row_sum != 0)
    D_inv = sp.diags(row_sum_inv)
    out = D_inv.dot(matrix).tocsr()
    return out


# https://github.com/mjDelta/relation-gcn-pytorch/blob/master/train.py
def to_sparse_tensor(sparse_array):
    if len(sp.find(sparse_array)[-1]) > 0:
        v = torch.FloatTensor(sp.find(sparse_array)[-1])
        i = torch.LongTensor(sparse_array.nonzero())
        shape = sparse_array.shape
        sparse_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    else:
        sparse_tensor = torch.empty([sparse_array.shape[0], sparse_array.shape[1]], layout=torch.sparse_coo)
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
