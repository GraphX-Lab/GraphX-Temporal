import numpy as np
import torch 

__all__ = ['remove_outbound_edge']

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

