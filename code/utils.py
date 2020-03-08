# -*- coding: utf-8 -*-
"""
"""

import math
import torch
import numpy as np
from scipy import sparse


def create_adjacency_matrix(graph):
    
    """
    Sparse adjacency matrix
    
    Parameters
    ----------
    graph : Networkx graph
          training graph
          
    Return
    ----------
    A : Adjacency matrix
    """
    index_1 = [int(edge[0]) for edge in graph.edges()] + [int(edge[1]) for edge in graph.edges()]
    index_2 = [int(edge[1]) for edge in graph.edges()] + [int(edge[0]) for edge in graph.edges()]
    values = [1 for edge in index_1]
    node_count = max(max(index_1)+1, max(index_2)+1)
    A = sparse.coo_matrix((values, (index_1, index_2)), shape=(node_count, node_count), dtype=np.float32)
    return A

def normalize_adjacency_matrix(A, I):
    
    """
    Normalized adjacency matrix
    
    Parameters
    ----------
    A : Adjacency matrix
    I :  Identity matrix
          
    Return
    ----------
    A_tilde_hat : Normalized adjacency matrix
    """
    
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sparse.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat

def create_propagator_matrix(graph, alpha, model):
    
    """
    Apropagation matrix
    
    Parameters
    ----------
    graph : Networkx graph
    alpha :  Teleporting parameter
    model: Propagation model      
    Return
    ----------
    propagator : propagator matrix
    """
    
    A = create_adjacency_matrix(graph)
    I = sparse.eye(A.shape[0])
    A_tilde_hat = normalize_adjacency_matrix(A, I)
    if model == "exact":
        propagator = (I-(1-alpha)*A_tilde_hat).todense()
        propagator = alpha*torch.inverse(torch.FloatTensor(propagator))
    else:
        propagator = dict()
        A_tilde_hat = sparse.coo_matrix(A_tilde_hat)
        indices = np.concatenate([A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1).T
        propagator["indices"] = torch.LongTensor(indices)
        propagator["values"] = torch.FloatTensor(A_tilde_hat.data)
    return propagator


def uniform(size, tensor):
    
    """
    Weight initialization : uniform
    
    Parameters
    ----------
    size : Tensor size
    tensor : Initialized tensor
    """
    
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def create_alias_table(area_ratio):
    
    """
    
    Parameters
    ----------
    area_ration 
      
    Return
    ----------
    accept,alias
    
    """
    
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - \
            (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias


def alias_sample(accept, alias):
    
    """
    
    Parameters
    ----------
    accept
    alias
      
    Return
    ----------
    sample index
    
    """
    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]
    
def partition_num(num, workers):
    
    """
    Parameters
    ----------
    num
    workers
      
    """
    
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers] 