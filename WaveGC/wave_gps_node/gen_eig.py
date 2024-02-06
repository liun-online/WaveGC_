from torch_geometric.datasets import (CoraFull,Amazon, Actor, GNNBenchmarkDataset, Planetoid,
                                      TUDataset, WebKB, WikipediaNetwork, ZINC, Coauthor)

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import eigvals
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj, scatter)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from ogb.nodeproppred import PygNodePropPredDataset


dataset_dir = './dataset/ogbn_arxiv'
print(dataset_dir)

dataset = PygNodePropPredDataset(name='ogbn-arxiv')
data = dataset.get(0)
N=data.num_nodes
print(N)
laplacian_norm_type = 'sym'
undir_edge_index = to_undirected(data.edge_index)

L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, normalization=laplacian_norm_type,
                           num_nodes=N)
        )
print(L.shape)
evals, evects = np.linalg.eigh(L.toarray())
print('Finish')
np.save(dataset_dir+'/'+"evals.npy", evals)
np.save(dataset_dir+'/'+"evects.npy", evects)

