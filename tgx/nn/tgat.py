from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, PositionalEncoding, TemporalEncoding

from tgx.utils.functions import to_neighbor_list


class MergeLayer(nn.Module):

    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int):
        """
        Merge Layer to merge two inputs via: input_dim1 + input_dim2 -> hidden_dim -> output_dim.
        :param input_dim1: int, dimension of first input
        :param input_dim2: int, dimension of the second input
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        """
        merge and project the inputs
        :param input_1: Tensor, shape (*, input_dim1)
        :param input_2: Tensor, shape (*, input_dim2)
        :return:
        """
        # Tensor, shape (*, input_dim1 + input_dim2)
        x = torch.cat([input_1, input_2], dim=1)
        # Tensor, shape (*, output_dim)
        h = self.fc2(self.act(self.fc1(x)))
        return h

class MultiHeadAttention(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int,
                 num_heads: int = 2, dropout: float = 0.1):
        """
        Multi-head Attention module.
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim: int, dimension of time features (time encodings)
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(MultiHeadAttention, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        assert self.query_dim % num_heads == 0, "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"

        self.head_dim = self.query_dim // num_heads

        self.query_projection = nn.Linear(self.query_dim, num_heads * self.head_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)
        self.value_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)

        self.scaling_factor = self.head_dim ** -0.5

        self.layer_norm = nn.LayerNorm(self.query_dim)

        self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, node_time_features: torch.Tensor, neighbor_node_features: torch.Tensor,
                neighbor_node_time_features: torch.Tensor, neighbor_node_edge_features: torch.Tensor, neighbor_masks: np.ndarray):
        """
        temporal attention forward process
        :param node_features: Tensor, shape (batch_size, node_feat_dim)
        :param node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
        :param neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        :param neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        :param neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        :param neighbor_masks: ndarray, shape (batch_size, num_neighbors), used to create mask of neighbors for nodes in the batch
        :return:
        """
        # Tensor, shape (batch_size, 1, node_feat_dim)
        node_features = torch.unsqueeze(node_features, dim=1)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        # shape (batch_size, 1, num_heads, self.head_dim)
        query = self.query_projection(query).reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)

        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
        key = value = torch.cat([neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features], dim=2)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        key = self.key_projection(key).reshape(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        value = self.value_projection(value).reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        key = key.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        value = value.permute(0, 2, 1, 3)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention = torch.einsum('bhld,bhnd->bhln', query, key)
        attention = attention * self.scaling_factor

        # Tensor, shape (batch_size, 1, num_neighbors)
        if isinstance(neighbor_masks, np.ndarray):
            attention_mask = torch.from_numpy(neighbor_masks).to(node_features.device).unsqueeze(dim=1)
        else:
            attention_mask = neighbor_masks.unsqueeze(dim=1)
        attention_mask = attention_mask == 0
        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        attention_mask = torch.stack([attention_mask for _ in range(self.num_heads)], dim=1)

        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        # note that if a node has no valid neighbor (whose neighbor_masks are all zero), directly set the masks to -np.inf will make the
        # attention scores after softmax be nan. Therefore, we choose a very large negative number (-1e10 following TGAT) instead of -np.inf to tackle this case
        attention = attention.masked_fill(attention_mask, -1e10)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention_scores = self.dropout(torch.softmax(attention, dim=-1))

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        attention_output = torch.einsum('bhln,bhnd->bhld', attention_scores, value)

        # Tensor, shape (batch_size, 1, num_heads * self.head_dim), where num_heads * self.head_dim is equal to node_feat_dim + time_feat_dim
        attention_output = attention_output.permute(0, 2, 1, 3).flatten(start_dim=2)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.dropout(self.residual_fc(attention_output))

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.layer_norm(output + residual)

        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        output = output.squeeze(dim=1)
        # Tensor, shape (batch_size, num_heads, num_neighbors)
        attention_scores = attention_scores.squeeze(dim=2)

        return output, attention_scores


class TGATConv(MessagePassing):
    """
    Temporal Graph Attention Layer (TGAT) as described in the paper:
    "Inductive Representation Learning on Temporal Graphs" (https://arxiv.org/abs/2002.07962)

    This layer extends the traditional graph attention mechanism by incorporating temporal information
    into the attention computation, allowing it to effectively model dynamic graphs where edges have timestamps.

    """

    def __init__(
        self,
        hidden_channels,
        edge_dim=0,
        num_heads=1,
        dropout=0.1,
        output_channels=None,
        time_enc:Optional[Literal['time', 'pos']]=None,
        # message passing parameters
        **kwargs,
    ):
        super().__init__(
            **kwargs
        )
        if output_channels is None:
            output_channels = hidden_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.time_feat_dim  = hidden_channels if time_enc is not None else 0
        if time_enc == 'time':
            self.time_encoder = TemporalEncoding(hidden_channels)
        elif time_enc == 'pos':
            self.time_encoder = PositionalEncoding(hidden_channels)
        else:
            self.time_encoder = None

        self.attn = MultiHeadAttention(
            node_feat_dim=hidden_channels,
            edge_feat_dim=edge_dim,
            time_feat_dim=self.time_feat_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.merge = MergeLayer(
            input_dim1=hidden_channels + self.time_feat_dim,
            input_dim2=hidden_channels,
            hidden_dim=hidden_channels,
            output_dim=output_channels,
        )
    
    def compute_node_temporal_embeddings(
            self,
            x_i,
            time_i,
            x_j,
            edge_attr_j,
            time_j,
            msk = None,
        ):
        if self.time_encoder is not None:
            time_enc_i = self.time_encoder(time_i.float()).reshape(time_i.size(0), time_i.size(1), -1)
            time_enc_j = self.time_encoder(time_j.float()).reshape(time_j.size(0), time_j.size(1), -1)
        else:
            time_enc_i = torch.zeros_like(x_i).float()
            time_enc_j = torch.zeros_like(x_j).float()
        if msk is None:
            msk =torch.ones((x_i.size(0), x_j.size(1)), dtype=torch.bool, device=x_i.device)
        out, w = self.attn(
            x_i,
            time_enc_i,
            x_j,
            time_enc_j,
            edge_attr_j,
            msk,
        )
        out = self.merge(out, x_i)
        return out


    def forward(self, x, edge_index, edge_attr, edge_time):
        dst_nodes, edge_ids, neighbors = to_neighbor_list(edge_index)
        x_i = x[dst_nodes]
        x_j = x[neighbors]
        time_i  = torch.zeros((x_i.size(0), 1), device=x_i.device)
        time_j = edge_time[edge_ids]
        edge_attr_j = edge_attr[edge_ids]
        out = self.compute_node_temporal_embeddings(
            x_i,
            time_i,
            x_j,
            edge_attr_j,
            time_j,
        )
        return out