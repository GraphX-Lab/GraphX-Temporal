"""Base temporal data loader with common functionality."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from os import PathLike
from pathlib import Path
import numpy as np
import pandas as pd


class TemporalDatasetLoader(ABC):
    """
    Abstract base class for temporal graph data loaders.

    Provides common functionality for loading and processing temporal graph data.
    """

    def __init__(
        self,
        dataset_path: PathLike = Path("./data/"),
        dataset_name: str = 'default',
        seed: int = 2020
    ):
        """
        Initialize temporal data loader.

        Args:
            dataset_path: Path to dataset files
            dataset_name: Name of the dataset
            seed: Random seed for reproducibility
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_name
        self.seed = seed

    @abstractmethod
    def load(self):
        """Load and process the dataset. Must be implemented by subclasses."""
        pass

    def load_raw_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load raw temporal graph data from files.

        Returns:
            Tuple containing (sources, destinations, edge_idxs, labels, timestamps, edge_features)
        """
        # Load temporal edges - support both subdirectory structure and flat structure
        csv_path = self.dataset_path / self.dataset_name / f"ml_{self.dataset_name}.csv"
        edge_feat_path = self.dataset_path / self.dataset_name / f"ml_{self.dataset_name}.npy"
        node_feat_path = self.dataset_path / self.dataset_name / f"ml_{self.dataset_name}_node.npy"

        try:
            graph_df = pd.read_csv(csv_path)
            edge_features = np.load(edge_feat_path)[1:]
            node_features = np.load(node_feat_path)
        except FileNotFoundError:
            # Fallback to flat structure without subdirectory
            csv_path = self.dataset_path / f"ml_{self.dataset_name}.csv"
            edge_feat_path = self.dataset_path / f"ml_{self.dataset_name}.npy"
            node_feat_path = self.dataset_path / f"ml_{self.dataset_name}_node.npy"

            graph_df = pd.read_csv(csv_path)
            edge_features = np.load(edge_feat_path)
            node_features = np.load(node_feat_path)
        max_node_id = max(graph_df.u.max(), graph_df.i.max())

        if node_features.shape[0] < max_node_id + 1:
            print(f"Pad node features from {node_features.shape[0]} to {max_node_id + 1}")
            npad = np.zeros((max_node_id + 1 - node_features.shape[0], node_features.shape[1]))
            node_features = np.vstack((node_features, npad))

        # Extract arrays
        sources = graph_df.u.values
        destinations = graph_df.i.values
        edge_idxs = graph_df.idx.values
        labels = graph_df.label.values
        timestamps = graph_df.ts.values

        return sources, destinations, edge_idxs, labels, timestamps, edge_features, node_features

    def compute_temporal_splits(
        self,
        timestamps: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[float, float]:
        """
        Compute time thresholds for temporal splitting.

        Args:
            timestamps: Array of timestamps
            train_ratio: Training data ratio
            val_ratio: Validation data ratio

        Returns:
            Tuple of (val_time, test_time)
        """
        val_time = np.quantile(timestamps, train_ratio)
        test_time = np.quantile(timestamps, train_ratio + val_ratio)

        return val_time, test_time

    def create_temporal_masks(
        self,
        timestamps: np.ndarray,
        val_time: float,
        test_time: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create temporal split masks.

        Args:
            timestamps: Array of timestamps
            val_time: Validation time threshold
            test_time: Test time threshold

        Returns:
            Tuple of (train_mask, val_mask, test_mask)
        """
        train_mask = timestamps <= val_time
        val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
        test_mask = timestamps > test_time

        return train_mask, val_mask, test_mask

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the dataset.

        Returns:
            Dictionary containing dataset statistics
        """
        sources, destinations, edge_idxs, labels, timestamps, edge_features, node_features = self.load_raw_data()

        # Basic statistics
        num_nodes = max(np.max(sources), np.max(destinations)) + 1
        num_edges = len(sources)
        time_span = np.max(timestamps) - np.min(timestamps)
        unique_nodes = len(np.unique(np.concatenate([sources, destinations])))

        # Edge type statistics (if labels exist)
        edge_type_counts = {}
        if labels is not None and len(labels) > 0:
            unique_labels, counts = np.unique(labels, return_counts=True)
            edge_type_counts = dict(zip(unique_labels.tolist(), counts.tolist()))

        # Feature statistics
        edge_feat_dim = edge_features.shape[1] if edge_features is not None else 0
        node_feat_dim = node_features.shape[1] if node_features is not None else 0

        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'unique_nodes': unique_nodes,
            'time_span': float(time_span),
            'time_range': (float(np.min(timestamps)), float(np.max(timestamps))),
            'edge_type_counts': edge_type_counts,
            'edge_feature_dim': edge_feat_dim,
            'node_feature_dim': node_feat_dim,
            'num_edge_types': len(edge_type_counts) if edge_type_counts else 1
        }

    def validate_data(
        self,
        sources: np.ndarray,
        destinations: np.ndarray,
        timestamps: np.ndarray,
        edge_features: Optional[np.ndarray] = None,
        node_features: Optional[np.ndarray] = None
    ) -> bool:
        """
        Validate loaded data for consistency.

        Args:
            sources: Source node indices
            destinations: Destination node indices
            timestamps: Edge timestamps
            edge_features: Edge features (optional)
            node_features: Node features (optional)

        Returns:
            True if data is valid, raises exception otherwise
        """
        # Check array shapes
        if len(sources) != len(destinations) or len(sources) != len(timestamps):
            raise ValueError("Inconsistent array lengths for sources, destinations, and timestamps")

        # Check for negative node indices
        if np.any(sources < 0) or np.any(destinations < 0):
            raise ValueError("Node indices must be non-negative")

        # Check temporal ordering
        if len(np.unique(timestamps)) > 1 and not np.all(timestamps == np.sort(timestamps)):
            print("Warning: Timestamps are not sorted")

        # Check feature consistency
        if edge_features is not None and len(edge_features) != len(sources):
            raise ValueError("Number of edge features must match number of edges")

        if node_features is not None:
            max_node_id = max(np.max(sources), np.max(destinations))
            if len(node_features) <= max_node_id:
                raise ValueError(f"Number of node features ({len(node_features)}) "
                               f"must exceed maximum node ID ({max_node_id})")

        return True