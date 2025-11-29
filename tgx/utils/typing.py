from typing import Any, NotRequired, Optional, Protocol, TypedDict

import torch
from torch_geometric.data import Data
from torch_geometric.typing import Adj, PairTensor

# NodeEmbeddingsType = torch.Tensor | PairTensor

class ModelOutputs(TypedDict):
    """TypedDict for model outputs with optional "node_embeddings" and "edge_embeddings" fields.
    node_embeddings: 
        - torch.Tensor: Node embeddings tensor of shape (num_nodes, node_feat_dim).
        - PairTensor: PairTensor containing source and destination node embeddings.
    edge_embeddings: torch.Tensor, shape (num_edges, edge_feat_dim).
    edge_index: Dense adjacency matrix with shape (num_nodes, num_nodes) or Sparse edge index with shape (2, num_edges).
    data: Data, original or processed data batch.
    """

    node_embeddings: NotRequired[torch.Tensor|PairTensor]
    edge_embeddings: NotRequired[torch.Tensor]
    edge_index: NotRequired[Adj]
    data: NotRequired[Data]
    
    __extra_items__: Any


class DynamicGNNProtocal(Protocol):
    """Protocol for dynamic GNN models."""
    def compute_src_dst_node_temporal_embeddings():...

    def compute_node_temporal_embeddings():...
    
