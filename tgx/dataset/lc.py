"""Link classification data loader for temporal graphs."""

from typing import Optional, Tuple
from os import PathLike
from pathlib import Path
import numpy as np
import pandas as pd
import random

from ..data.temporal_data import Data
from .temporal_dataset import TemporalDatasetLoader


class LCDatasetLoader(TemporalDatasetLoader):
    """
    Link classification data loader for temporal graphs.

    Handles loading and preprocessing of temporal graph data for link classification tasks.
    """

    def __init__(
        self,
        dataset_path: PathLike = Path("./data/"),
        dataset_name: str = 'mooc',
        use_validation: bool = True,
        train_ratio: Optional[float] = None,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 2020
    ):
        """
        Initialize link classification data loader.

        Args:
            dataset_path: Path to dataset files
            dataset_name: Name of the dataset
            use_validation: Whether to use validation split
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed for reproducibility
        """
        super().__init__(dataset_path, dataset_name, seed)
        self.use_validation = use_validation
        if train_ratio is None:
            train_ratio = round(1 - val_ratio - test_ratio, 5)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        random.seed(seed)

    def load(self) -> Tuple[
        Data,        # full_data
        np.ndarray,  # node_features
        np.ndarray,  # edge_features
        Data,        # train_data
        Data,        # val_data
        Data         # test_data
    ]:
        """
        Load and process temporal graph data for node classification.

        Returns:
            Tuple containing data and features
        """
        # Load raw data using parent class method
        sources, destinations, edge_idxs, labels, timestamps, edge_features, node_features = self.load_raw_data()
        graph_df = pd.DataFrame({
            'u': sources,
            'i': destinations,
            'idx': edge_idxs,
            'label': labels,
            'ts': timestamps
        })
        train_start = round(1 - self.train_ratio - self.val_ratio - self.test_ratio, 5)
        train_end = 1 - self.val_ratio - self.test_ratio
        val_end = 1 - self.test_ratio


        # Compute time thresholds based on quantiles
        train_start_time, val_time, test_time = list(np.quantile(graph_df.ts, [train_start, train_end, val_end]))

        # Extract arrays
        sources = graph_df.u.values
        destinations = graph_df.i.values
        edge_idxs = graph_df.idx.values
        labels = graph_df.label.values
        timestamps = graph_df.ts.values

        # Create full data object
        full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

        # Create train/val/test masks based on time
        if self.use_validation:
            train_mask = np.logical_and(train_start_time < timestamps, timestamps <= val_time)
            val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
        else:
            train_mask = np.logical_and(train_start_time < timestamps, timestamps <= test_time)
            val_mask = test_mask = timestamps > test_time

        test_mask = timestamps > test_time

        # Create data splits
        train_data = Data(
            sources[train_mask], destinations[train_mask], timestamps[train_mask],
            edge_idxs[train_mask], labels[train_mask]
        )

        val_data = Data(
            sources[val_mask], destinations[val_mask], timestamps[val_mask],
            edge_idxs[val_mask], labels[val_mask]
        )

        test_data = Data(
            sources[test_mask], destinations[test_mask], timestamps[test_mask],
            edge_idxs[test_mask], labels[test_mask]
        )

        return full_data, node_features, edge_features, train_data, val_data, test_data

    def get_edge_labels(self) -> np.ndarray:
        """
        Get edge labels for link classification task.

        Returns:
            Array of edge labels (0/1 for binary classification)
        """
        # Edge labels are already loaded in the Data objects
        # This method is kept for compatibility but not needed for LC
        return None  # Edge labels are accessed through Data.labels

    def get_train_val_test_nodes(
        self,
        full_data: Data,
        val_ratio: float = None,
        test_ratio: float = None,
        train_ratio: float = 0.6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split nodes into train/validation/test sets.

        Args:
            full_data: Full temporal graph data
            val_ratio: Validation ratio (uses class default if None)
            test_ratio: Test ratio (uses class default if None)
            train_ratio: Training ratio

        Returns:
            Tuple of (train_nodes, val_nodes, test_nodes)
        """
        if val_ratio is None:
            val_ratio = self.val_ratio
        if test_ratio is None:
            test_ratio = self.test_ratio

        # Get all unique nodes
        all_nodes = np.unique(np.concatenate([full_data.sources, full_data.destinations]))
        np.random.shuffle(all_nodes)

        # Calculate split sizes
        total_nodes = len(all_nodes)
        train_size = int(train_ratio * total_nodes)
        val_size = int(val_ratio * total_nodes)

        # Split nodes
        train_nodes = all_nodes[:train_size]
        val_nodes = all_nodes[train_size:train_size + val_size]
        test_nodes = all_nodes[train_size + val_size:]

        return train_nodes, val_nodes, test_nodes