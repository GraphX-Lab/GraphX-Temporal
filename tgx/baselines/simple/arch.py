"""
SimpleGNN: Simple Graph Neural Network for Temporal Graph Learning

This model uses a GNN as backbone and can be used with task-specific evaluators
for both link prediction and node classification tasks.
It can optionally use timestamp encoding as edge features if supported.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TemporalEncoding, MLP
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar

from tgx.models.temporal_model import TemporalModel
from tgx.utils.typing import ModelOutputs

__all__ = ["SimpleGNNModel", "MLPModel"]


class MLPModel(TemporalModel):
    """
    Multi-layer perceptron for node classification and link prediction tasks.

    This model uses a simple MLP backbone that can work with node features
    or learned node embeddings for temporal graph learning tasks.
    """

    model_name = "MLP"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        use_node_embedding: bool = True,
        max_nodes: int = 10000,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.use_node_embedding = use_node_embedding
        self.max_nodes = max_nodes

        # Build MLP layers
        layers = []
        in_dim = input_dim

        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        # Output layer
        if output_dim is not None:
            layers.append(nn.Linear(hidden_dim, output_dim))
            in_dim = output_dim

        self.mlp = nn.Sequential(*layers)

        if use_node_embedding:
            self.node_embedding = nn.Embedding(max_nodes, input_dim)

        # Dropout and activation
        self.dropout_layer = nn.Dropout(dropout)

        # Activation function
        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "gelu":
            self.activation_fn = nn.GELU()
        elif activation == "tanh":
            self.activation_fn = nn.Tanh()
        else:
            self.activation_fn = nn.ReLU()

    def forward(self, batch: Data) -> ModelOutputs:
        """Forward pass of the MLP model."""
        # Handle node features
        if hasattr(batch, "x") and batch.x is not None:
            x = batch.x
            # Ensure feature dimension matches input_dim
            if x.size(-1) < self.input_dim:
                # Pad features if needed
                padding_size = self.input_dim - x.size(-1)
                padding = torch.zeros(
                    x.size(0), padding_size, device=x.device, dtype=x.dtype
                )
                x = torch.cat([x, padding], dim=-1)
            elif x.size(-1) > self.input_dim:
                # Truncate features if needed
                x = x[:, : self.input_dim]
        else:
            # Use node IDs as embeddings
            if hasattr(batch, "num_nodes") and batch.num_nodes is not None:
                node_ids = torch.arange(batch.num_nodes, device=batch.edge_index.device)
            else:
                node_ids = batch.edge_index.max() + 1

            if self.use_node_embedding:
                x = self.node_embedding(node_ids)
            else:
                # Create one-hot encodings if no embedding layer
                x = F.one_hot(node_ids, num_classes=self.input_dim).float()

        # Apply MLP layers with activation and dropout (except for last layer)
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i < len(self.mlp) - 1:  # Don't apply activation/dropout to last layer
                x = self.activation_fn(x)
                x = self.dropout_layer(x)

        # Handle different task types
        label_edge_index = getattr(batch, "edge_label_index", batch.edge_index)
        if label_edge_index.numel() > 0:
            # Use source node embeddings to predict each edge
            row = label_edge_index[0]
            edge_embeddings = x[row]

        return {
            "node_embeddings": x,
            "edge_embeddings": edge_embeddings,
        }


class SimpleGNNModel(TemporalModel):
    """
    Simple Graph Neural Network for temporal graph learning.

    This model provides a basic GNN backbone that can be used with task-specific
    evaluators for both link prediction and node classification tasks.
    It can optionally use time encoding as edge features.

    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for GNN
        gnn_type (str): Type of GNN ('gcn', 'gat', 'sage')
        num_layers (int): Number of GNN layers
        dropout (float): Dropout rate
        use_time_encoding (bool): Whether to use timestamp encoding
        time_dim (int): Dimension for time encoding
        task (str): Task type ('link_prediction', 'node_classification')
        **kwargs: Additional arguments passed to TemporalModel
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        gnn_type: str = "gcn",
        num_layers: int = 2,
        dropout: float = 0.1,
        use_time_encoding: bool = True,
        time_dim: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type.lower()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_time_encoding = use_time_encoding

        # Time encoder
        if use_time_encoding:
            self.time_encoder = TemporalEncoding(time_dim)
            # Edge feature dimension will be input_dim + time_dim for GNNs that support edge_attr
            edge_feature_dim = time_dim
        else:
            edge_feature_dim = 0

        # Node embedding for nodes without features
        # self.node_embedding = nn.Embedding(200000, input_dim)  # Assume max 200000 nodes

        self.mlp = MLP([input_dim, input_dim, input_dim])
        # GNN backbone
        self.gnn_layers = nn.ModuleList()

        if self.gnn_type == "gcn":
            self.gnn_layers.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))

        elif self.gnn_type == "gat":
            self.gnn_layers.append(
                GATConv(input_dim, hidden_dim, heads=4, concat=False)
            )
            for _ in range(num_layers - 1):
                self.gnn_layers.append(
                    GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
                )

        elif self.gnn_type == "sage":
            self.gnn_layers.append(SAGEConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))

        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, batch: Data) -> ModelOutputs:
        """Forward pass of the model."""

        # if hasattr(batch, "num_nodes") and batch.num_nodes is not None:
        #     node_ids = torch.arange(batch.num_nodes, device=batch.edge_index.device)
        # else:
        #     node_ids = batch.edge_index.max() + 1
        # node_emb = self.node_embedding(node_ids)

        x = batch.x
        if x.size(-1) < self.input_dim:
            # Pad features if needed
            padding = self.input_dim - x.size(-1)
            x = F.pad(x, (0, padding))
        elif x.size(-1) > self.input_dim:
            # Truncate features if needed
            x = x[:, : self.input_dim]
        x = torch.layer_norm(self.mlp(x) + x, normalized_shape=(x.size(-1),))

        # x += node_emb

        # Edge features (time encoding)
        edge_attr = None
        if self.use_time_encoding and hasattr(batch, "t") and batch.t is not None:
            edge_attr = self.time_encoder(batch.t)

        # Ensure edge_index is valid: must be LongTensor and within bounds
        edge_index = batch.edge_index.long()
        if edge_index.numel() > 0:
            # Clamp indices to valid range to prevent CUDA errors
            max_node_idx = x.size(0) - 1
            edge_index = torch.clamp(edge_index, 0, max_node_idx)

        # Apply GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            # Note: Some GNN layers support edge_attr, others don't
            try:
                if (
                    edge_attr is not None
                    and hasattr(gnn_layer, "forward")
                    and "edge_attr" in gnn_layer.forward.__code__.co_varnames
                ):
                    x = gnn_layer(x, edge_index, edge_attr=edge_attr)
                else:
                    x = gnn_layer(x, edge_index)
            except Exception as e:
                # Fallback to basic forward without edge_attr
                print(f"GNN layer {i} failed with edge_attr, falling back: {e}")
                x = gnn_layer(x, edge_index)

            if i < len(self.gnn_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)

        # Return node embeddings and edge information for task-specific evaluators
        return {
            "node_embeddings": x,
            "edge_index": batch.edge_index,
            "edge_attr": edge_attr,
            "batch": batch,
        }

    @property
    def model_name(self) -> str:
        return f"SimpleGNN-{self.gnn_type.upper()}"

    def configure_callbacks(self):
        task = "lp" if "LP" in self.trainer.datamodule.__class__.__name__ else "nc"
        cbs = [
            ModelCheckpoint(
                f"./checkpoints/simple/gcn/{self.trainer.datamodule.dataset}/{task}/",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                filename="best",
                save_last=True,
                verbose=False
            ),
            EarlyStopping(monitor="val/loss", patience=10, mode="min", verbose=False),
            RichProgressBar(leave=False),
        ]
        return cbs

    def create_edge_embeddings(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Create edge embeddings by summing source and target node embeddings."""
        source_embeddings = node_embeddings[edge_index[0]]
        target_embeddings = node_embeddings[edge_index[1]]
        edge_embeddings = source_embeddings + target_embeddings
        return edge_embeddings
