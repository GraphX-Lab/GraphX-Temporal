"""Temporal graph data structures."""

from typing import Dict, Optional
import numpy as np
import torch


class Data:
    """
    Temporal graph data container.

    Stores temporal graph information including sources, destinations,
    timestamps, edge indices, and labels.
    """

    def __init__(
        self,
        sources: np.ndarray,
        destinations: np.ndarray,
        timestamps: np.ndarray,
        edge_idx: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        edge_feats: Optional[np.ndarray] = None,
        node_feats: Optional[np.ndarray] = None,
    ):
        """
        Initialize temporal graph data.

        Args:
            sources: Source node indices
            destinations: Destination node indices
            timestamps: Edge timestamps
            edge_idx: Edge indices (optional)
            labels: Edge labels (optional)
            edge_feats: Edge features (optional)
            node_feats: Node features (optional)
        """
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idx = edge_idx
        self.labels = labels
        self.edge_feats = edge_feats
        self.node_feats = node_feats

        self.num_edges = len(sources)
        self.num_nodes = (
            max(max(sources), max(destinations)) + 1 if len(sources) > 0 else 0
        )

    def __len__(self) -> int:
        """Return number of edges."""
        return self.num_edges

    def __repr__(self) -> str:
        """String representation of the data."""
        return (
            f"Data(num_edges={self.num_edges}, "
            f"num_nodes={self.num_nodes}, "
            f"time_range=({np.min(self.timestamps):.2f}, "
            f"{np.max(self.timestamps):.2f})"
        )

    def apply_mask(self, mask: np.ndarray) -> "Data":
        """
        Apply boolean mask to create a new Data object.

        Args:
            mask: Boolean mask to apply

        Returns:
            New Data object with filtered edges
        """
        new_data = Data(
            sources=self.sources[mask],
            destinations=self.destinations[mask],
            timestamps=self.timestamps[mask],
        )

        if self.edge_idx is not None:
            new_data.edge_idx = self.edge_idx[mask]
        if self.labels is not None:
            new_data.labels = self.labels[mask]
        if self.edge_feats is not None:
            new_data.edge_feats = self.edge_feats[mask]
        if self.node_feats is not None:
            new_data.node_feats = self.node_feats

        return new_data

    def get_time_mask(self, start_time: float, end_time: float) -> np.ndarray:
        """
        Get boolean mask for edges within time range.

        Args:
            start_time: Start of time range (inclusive)
            end_time: End of time range (exclusive)

        Returns:
            Boolean mask for edges in time range
        """
        return (self.timestamps >= start_time) & (self.timestamps < end_time)

    def get_temporal_slice(self, start_time: float, end_time: float) -> "Data":
        """
        Get temporal slice of the data.

        Args:
            start_time: Start of time range (inclusive)
            end_time: End of time range (exclusive)

        Returns:
            New Data object with temporal slice
        """
        mask = self.get_time_mask(start_time, end_time)
        return self.apply_mask(mask)

    def to_tensor_dict(self) -> Dict[str, torch.Tensor]:
        """
        Convert data to a dictionary of numpy arrays.

        Returns:
            Dictionary with keys:
                - edge_index: LongTensor of shape [2, num_edges]
                - timestamps: FloatTensor of shape [num_edges]
                - edge_idx, 'edge_labels': LongTensor of shape [num_edges] if applicable
                - edge_attr: FloatTensor of shape [num_edges, num_edge_features] if applicable
                - x: FloatTensor of shape [num_nodes, num_node_features] if applicable

        """
        data_dict = {
            "edge_index": torch.stack(
                [
                    torch.from_numpy(self.sources).long(),
                    torch.from_numpy(self.destinations).long(),
                ],
                dim=0,
            ),
            "edge_time": torch.from_numpy(self.timestamps).long(),
        }
        if self.edge_idx is not None:
            data_dict["edge_idx"] = torch.from_numpy(self.edge_idx).long()
        else:
            data_dict["edge_idx"] = torch.arange(self.num_edges).long()

        if self.labels is not None:
            data_dict["edge_labels"] = torch.from_numpy(self.labels).long()
        else:
            data_dict["edge_labels"] = None

        if self.edge_feats is not None:
            data_dict["edge_attr"] = torch.from_numpy(self.edge_feats).float()
        else:
            data_dict["edge_attr"] = None

        if self.node_feats is not None:
            data_dict["x"] = torch.from_numpy(self.node_feats).float()
        else:
            data_dict["x"] = None

        return data_dict
