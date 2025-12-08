"""Link prediction data loader for temporal graphs."""

from typing import Tuple, Optional
from os import PathLike
from pathlib import Path
import numpy as np
import pandas as pd
import random

from ..data.temporal_data import Data
from .temporal_dataset import TemporalDatasetLoader


class LPDatasetLoader(TemporalDatasetLoader):
    """
    Link prediction data loader for temporal graphs.

    Handles loading and preprocessing of temporal graph data for link prediction tasks,
    with support for inductive evaluation using unseen nodes.
    """

    inductive_edge_types = ["new_old", "old_new", "new_new"]

    def __init__(
        self,
        dataset_path: PathLike = Path("./data"),
        dataset_name: str = "mooc",
        different_new_nodes_between_val_and_test: bool = False,
        randomize_features: bool = False,
        train_ratio: Optional[float] = None,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 2020,
    ):
        """
        Initialize link prediction data loader.

        Args:
            dataset_path: Path to dataset files
            dataset_name: Name of the dataset
            different_new_nodes_between_val_and_test: Whether to use different new nodes for val/test
            randomize_features: Whether to randomize node features
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed for reproducibility
        """
        super().__init__(dataset_path, dataset_name, seed)
        self.different_new_nodes_between_val_and_test = (
            different_new_nodes_between_val_and_test
        )
        self.randomize_features = randomize_features
        if train_ratio is None:
            train_ratio = round(1 - val_ratio - test_ratio, 5)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        # random.seed(seed)

    def load(
        self,
    ) -> Tuple[
        np.ndarray,  # node_features
        np.ndarray,  # edge_features
        Data,  # full_data
        Data,  # train_data
        Data,  # val_data
        Data,  # test_data
        Data,  # new_node_val_data
        Data,  # new_node_test_data
        Data,  # new_old_node_val_data
        Data,  # new_old_node_test_data
        Data,  # new_new_node_val_data
        Data,  # new_new_node_test_data
        # int,          # unseen_nodes_num
        set,  # new_test_node_set
    ]:
        """
        Load and process temporal graph data.

        Returns:
            Tuple containing node features, edge features, and various data splits
        """
        # Load raw data using parent class method
        (
            sources,
            destinations,
            edge_idxs,
            labels,
            timestamps,
            edge_features,
            node_features,
        ) = self.load_raw_data()

        if self.randomize_features:
            node_features = self.np_rng.random(node_features.shape)

        # Create DataFrame for easier manipulation
        graph_df = pd.DataFrame(
            {
                "u": sources,
                "i": destinations,
                "idx": edge_idxs,
                "label": labels,
                "ts": timestamps,
            }
        )

        # Extract arrays
        sources = graph_df.u.values
        destinations = graph_df.i.values
        edge_idxs = graph_df.idx.values
        labels = graph_df.label.values
        timestamps = graph_df.ts.values

        # Compute time thresholds
        train_start = round(1 - self.train_ratio - self.val_ratio - self.test_ratio, 5)

        train_end = 1 - self.val_ratio - self.test_ratio
        val_end = 1 - self.test_ratio
        train_start_time, val_time, test_time = list(
            np.quantile(timestamps, [train_start, train_end, val_end])
        )

        # Create full data object
        full_data = Data(
            sources, destinations, timestamps, edge_idxs, labels, edge_features
        )

        # Get all unique nodes
        node_set = set(sources) | set(destinations)
        n_total_unique_nodes = len(node_set)

        # Find test-time nodes for inductive evaluation
        test_time_mask = timestamps > val_time
        test_node_set = set(sources[test_time_mask]).union(
            set(destinations[test_time_mask])
        )

        # Sample new test nodes
        new_test_node_set = set(
            self.rng.sample(list(test_node_set), int(0.1 * n_total_unique_nodes))
        )

        # Create masks for new test nodes
        new_test_source_mask = np.array([src in new_test_node_set for src in sources])
        new_test_destination_mask = np.array(
            [dst in new_test_node_set for dst in destinations]
        )

        # Mask for observed edges (no new test nodes involved)
        observed_edges_mask = np.logical_and(
            ~new_test_source_mask, ~new_test_destination_mask
        )

        # Training mask (before val time, no new nodes)
        train_mask = np.logical_and.reduce(
            [train_start_time < timestamps, timestamps <= val_time, observed_edges_mask]
        )

        # Create training data
        train_data = Data(
            sources[train_mask],
            destinations[train_mask],
            timestamps[train_mask],
            edge_idxs[train_mask],
            labels[train_mask],
            edge_features[train_mask] if edge_features is not None else None,
        )
        before_val_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
        seen_data = Data(
            sources[before_val_mask],
            destinations[before_val_mask],
            timestamps[before_val_mask],
            edge_idxs[before_val_mask],
            labels[before_val_mask],
            edge_features[before_val_mask] if edge_features is not None else None,
        )

        # Define new node sets
        # train_node_set = set(train_data.sources).union(set(train_data.destinations))
        seen_node_set = set(seen_data.sources).union(set(seen_data.destinations))
        new_node_set = node_set - seen_node_set

        assert len(seen_node_set & new_test_node_set) == 0, (
            "Overlap between seen and new test nodes (unseen)"
        )

        # Validation and test masks
        val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
        test_mask = timestamps > test_time

        # Handle different new nodes for val/test
        if self.different_new_nodes_between_val_and_test:
            n_new_nodes = len(new_test_node_set) // 2
            val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
            test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

            # Edge masks for validation
            edge_contains_new_val_node_mask = np.array(
                [
                    (src in val_new_node_set or dst in val_new_node_set)
                    for src, dst in zip(sources, destinations)
                ]
            )
            edge_contains_new_test_node_mask = np.array(
                [
                    (src in test_new_node_set or dst in test_new_node_set)
                    for src, dst in zip(sources, destinations)
                ]
            )

            new_node_val_mask = np.logical_and(
                val_mask, edge_contains_new_val_node_mask
            )
            new_node_test_mask = np.logical_and(
                test_mask, edge_contains_new_test_node_mask
            )
        else:
            # Use same new node set for both val and test
            edge_contains_new_node_mask = np.array(
                [
                    (src in new_node_set or dst in new_node_set)
                    for src, dst in zip(sources, destinations)
                ]
            )
            new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
            new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

            # Separate new-new and new-old edges
            edge_contains_new_new_node_mask = np.array(
                [
                    (src in new_node_set and dst in new_node_set)
                    for src, dst in zip(sources, destinations)
                ]
            )
            edge_contains_new_old_node_mask = np.logical_and(
                edge_contains_new_node_mask,
                np.logical_not(edge_contains_new_new_node_mask),
            )

            new_new_node_val_mask = np.logical_and(
                val_mask, edge_contains_new_new_node_mask
            )
            new_new_node_test_mask = np.logical_and(
                test_mask, edge_contains_new_new_node_mask
            )
            new_old_node_val_mask = np.logical_and(
                val_mask, edge_contains_new_old_node_mask
            )
            new_old_node_test_mask = np.logical_and(
                test_mask, edge_contains_new_old_node_mask
            )

        # Create data splits
        val_data = Data(
            sources[val_mask],
            destinations[val_mask],
            timestamps[val_mask],
            edge_idxs[val_mask],
            labels[val_mask],
            edge_features[val_mask] if edge_features is not None else None,
        )
        test_data = Data(
            sources[test_mask],
            destinations[test_mask],
            timestamps[test_mask],
            edge_idxs[test_mask],
            labels[test_mask],
            edge_features[test_mask] if edge_features is not None else None,
        )

        # New node evaluation data
        new_node_val_data = Data(
            sources[new_node_val_mask],
            destinations[new_node_val_mask],
            timestamps[new_node_val_mask],
            edge_idxs[new_node_val_mask],
            labels[new_node_val_mask],
        )
        new_node_test_data = Data(
            sources[new_node_test_mask],
            destinations[new_node_test_mask],
            timestamps[new_node_test_mask],
            edge_idxs[new_node_test_mask],
            labels[new_node_test_mask],
        )

        if not self.different_new_nodes_between_val_and_test:
            new_old_node_val_data = Data(
                sources[new_old_node_val_mask],
                destinations[new_old_node_val_mask],
                timestamps[new_old_node_val_mask],
                edge_idxs[new_old_node_val_mask],
                labels[new_old_node_val_mask],
            )
            new_old_node_test_data = Data(
                sources[new_old_node_test_mask],
                destinations[new_old_node_test_mask],
                timestamps[new_old_node_test_mask],
                edge_idxs[new_old_node_test_mask],
                labels[new_old_node_test_mask],
            )
            new_new_node_val_data = Data(
                sources[new_new_node_val_mask],
                destinations[new_new_node_val_mask],
                timestamps[new_new_node_val_mask],
                edge_idxs[new_new_node_val_mask],
                labels[new_new_node_val_mask],
            )
            new_new_node_test_data = Data(
                sources[new_new_node_test_mask],
                destinations[new_new_node_test_mask],
                timestamps[new_new_node_test_mask],
                edge_idxs[new_new_node_test_mask],
                labels[new_new_node_test_mask],
            )
        else:
            # Dummy data for compatibility
            new_old_node_val_data = new_old_node_test_data = new_new_node_val_data = (
                new_new_node_test_data
            ) = None

        unseen_nodes_num = len(new_test_node_set)

        return (
            node_features,
            edge_features,
            full_data,
            train_data,
            val_data,
            test_data,
            new_node_val_data,
            new_node_test_data,
            new_old_node_val_data,
            new_old_node_test_data,
            new_new_node_val_data,
            new_new_node_test_data,
            new_test_node_set,
        )
