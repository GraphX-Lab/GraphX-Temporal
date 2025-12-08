from typing import Optional
import numpy as np
from sympy import I
import torch
from tgx.data import Data as TemporalData

__all__ = ["remove_outbound_edge", "to_neighbor_list"]


def remove_outbound_edge(x, edge_index, edge_attr=None):
    """
    Remove outbound edges from the graph according to x.size(0).
    Supports both numpy and torch tensors.
    """
    if isinstance(x, np.ndarray):
        return __remove_outbound_edge_np(x, edge_index, edge_attr)
    elif isinstance(x, torch.Tensor):
        return __remove_outbound_edge_torch(x, edge_index, edge_attr)
    else:
        raise TypeError("Unsupported type for x. Expected np.ndarray or torch.Tensor.")


def __remove_outbound_edge_np(x, edge_index, edge_attr=None):
    """
    Remove outbound edges from the graph according to x.size(0).
    """
    src, dst = edge_index
    mask = np.logical_or(src >= x.shape[0], dst >= x.shape[0])
    edge_index = edge_index[:, ~mask]
    if edge_attr is not None:
        edge_attr = edge_attr[~mask]
        return edge_index, edge_attr
    return edge_index


def __remove_outbound_edge_torch(x, edge_index, edge_attr=None):
    """
    Remove outbound edges from the graph according to x.size(0).
    """
    src, dst = edge_index
    mask = torch.logical_or(src >= x.size(0), dst >= x.size(0))
    edge_index = edge_index[:, ~mask]
    if edge_attr is not None:
        edge_attr = edge_attr[~mask]
        return edge_index, edge_attr
    return edge_index


def to_neighbor_list(edge_index):
    """
    Convert edge_index to neighbor list representation.
    """
    src, dst = edge_index
    dst_nodes = torch.unique(dst)
    node_index = {node.item(): i for i, node in enumerate(dst_nodes)}
    num_nodes = len(dst_nodes)
    neighbor_list = [[] for _ in range(num_nodes)]
    edge_ids = [[] for _ in range(num_nodes)]
    for eid, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
        d_i = node_index[d] 
        neighbor_list[d_i].append(s)
        edge_ids[d_i].append(eid)
    neighbors = torch.ones(num_nodes, max(len(nbrs) for nbrs in neighbor_list), dtype=torch.long, device=dst.device) * -1
    edge_ids_tensor = torch.ones(num_nodes, max(len(eids) for eids in edge_ids), dtype=torch.long, device=dst.device) * -1
    for i in range(num_nodes):
        nbrs = neighbor_list[i]
        eids = edge_ids[i]
        neighbors[i, :len(nbrs)] = torch.tensor(nbrs, dtype=torch.long)
        edge_ids_tensor[i, :len(eids)] = torch.tensor(eids, dtype=torch.long)
    # neighbors = torch.tensor(neighbor_list, dtype=torch.long)
    # edge_ids = torch.tensor(edge_ids, dtype=torch.long)
    edge_ids = edge_ids_tensor
    return dst_nodes, edge_ids, neighbors