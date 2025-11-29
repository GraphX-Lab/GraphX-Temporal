"""
DataModule for Link Prediction (LP) tasks.

This module provides specialized data loading and processing for link prediction
tasks using the benchtemp format with LPDatasetLoader.
"""

from pathlib import Path
from typing import List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling

from tgx.dataset.lp import LPDatasetLoader
from ..temporal_data import Data as TemporalData


class LPDataModule(pl.LightningDataModule):
    inductive_edge_types = ["new_old", "old_new", "new_new"]

    """
    PyTorch Lightning DataModule for link prediction tasks.

    This module handles loading and preparation of temporal graph datasets
    specifically for link prediction with proper train/val/test splitting
    and edge-level processing using LPDatasetLoader.

    Args:
        dataset (str): Dataset name ('mooc', 'reddit', 'wikipedia', etc.)
        data_dir (str): Directory containing data files
        batch_size (int): Batch size for training
        num_workers (int): Number of data loading workers
        pin_memory (bool): Whether to pin memory for faster GPU transfer
        shuffle_train (bool): Whether to shuffle training data
        negative_sampling_ratio (float): Ratio of negative edges to sample
        different_new_nodes_between_val_and_test (bool): Whether to use different new nodes
    """

    def __init__(
        self,
        dataset: str = "mooc",
        data_dir: str = "./datasets",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle_train: bool = True,
        negative_sampling_ratio: float = 1.0,
        different_new_nodes_between_val_and_test: bool = False,
        split_ratio: List[float] = [0.7, 0.15, 0.15],
        mode: str = "transductive",
        **kwargs,
    ):
        super().__init__()
        self.dataset = dataset
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.negative_sampling_ratio = negative_sampling_ratio
        self.different_new_nodes_between_val_and_test = (
            different_new_nodes_between_val_and_test
        )
        if len(split_ratio) == 2:
            split_ratio = [None] + split_ratio  # Train ratio will be inferred

        self.split_ratio = split_ratio

        # LPDatasetLoader will be initialized in setup() to avoid circular import
        self.dataset_loader = None
        self.mode = mode

        # Initialize data containers as None (will be created in setup)
        self.node_features = None
        self.edge_features = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.new_node_val_data = None
        self.new_node_test_data = None

        # Store basic dataset info
        self.num_nodes = None
        self.node_features_dim = None
        self.edge_features_dim = None

        self.np_rng = np.random.default_rng(42)

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing using LPDatasetLoader."""

        # Initialize LPDatasetLoader if not already done
        if self.dataset_loader is None:
            self.dataset_loader = LPDatasetLoader(
                dataset_path=str(self.data_dir) + "/",
                dataset_name=self.dataset,
                different_new_nodes_between_val_and_test=self.different_new_nodes_between_val_and_test,
                train_ratio=self.split_ratio[0],
                val_ratio=self.split_ratio[1],
                test_ratio=self.split_ratio[2],
            )

        # Load data using LPDatasetLoader
        (
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
            # unseen_nodes_num,
            new_test_node_set,
        ) = self.dataset_loader.load()

        # Store loaded data
        self.node_features = node_features
        self.edge_features = edge_features
        self.full_data: TemporalData = full_data
        self.train_data: TemporalData = train_data
        self.val_data: TemporalData = val_data
        self.test_data: TemporalData = test_data
        self.new_node_val_data: TemporalData = new_node_val_data
        self.new_node_test_data: TemporalData = new_node_test_data
        self.new_old_node_val_data: TemporalData = new_old_node_val_data
        self.new_old_node_test_data: TemporalData = new_old_node_test_data
        self.new_new_node_val_data: TemporalData = new_new_node_val_data
        self.new_new_node_test_data: TemporalData = new_new_node_test_data
        self.unseen_nodes = new_test_node_set
        self.unseen_nodes_num = len(new_test_node_set)

        # Inductive node info
        self.train_node_type_info = None
        self.val_node_type_info = None
        self.test_node_type_info = None

        # edge index dict for different message passing schemes
        self.message_edge_index_dict = {}

        # Create node type information for inductive evaluation
        if self.mode == "inductive":
            self._create_node_type_info()

        # Store dataset information
        self._store_dataset_info()

    def _create_node_type_info(self):
        """Create node type information for inductive evaluation."""
        import numpy as np

        new_nodes = self.unseen_nodes
        # Create node type mapping: True for new nodes, False for old nodes
        max_node_id = self.full_data.num_nodes - 1
        is_new_node = {
            node_id: (node_id in new_nodes) for node_id in range(max_node_id + 1)
        }

        # Store node type information for each dataset
        self.train_node_type_info = {
            "is_new_node": np.array(
                [
                    is_new_node[src] or is_new_node[dst]
                    for src, dst in zip(
                        self.train_data.sources, self.train_data.destinations
                    )
                ],
                dtype=bool,
            )
        }

        if self.val_data is not None:
            self.val_node_type_info = {
                "is_new_node": np.array(
                    [
                        is_new_node[src] or is_new_node[dst]
                        for src, dst in zip(
                            self.val_data.sources, self.val_data.destinations
                        )
                    ],
                    dtype=bool,
                )
            }
        else:
            self.val_node_type_info = None

        if self.test_data is not None:
            self.test_node_type_info = {
                "is_new_node": np.array(
                    [
                        is_new_node[src] or is_new_node[dst]
                        for src, dst in zip(
                            self.test_data.sources, self.test_data.destinations
                        )
                    ],
                    dtype=bool,
                )
            }
        else:
            self.test_node_type_info = None

        # Create additional node type info for different data splits
        if self.new_node_val_data is not None:
            self.new_node_val_type_info = {
                "is_new_node": np.array(
                    [
                        is_new_node[src] or is_new_node[dst]
                        for src, dst in zip(
                            self.new_node_val_data.sources,
                            self.new_node_val_data.destinations,
                        )
                    ],
                    dtype=bool,
                )
            }
        else:
            self.new_node_val_type_info = None

        if self.new_node_test_data is not None:
            self.new_node_test_type_info = {
                "is_new_node": np.array(
                    [
                        is_new_node[src] or is_new_node[dst]
                        for src, dst in zip(
                            self.new_node_test_data.sources,
                            self.new_node_test_data.destinations,
                        )
                    ],
                    dtype=bool,
                )
            }
        else:
            self.new_node_test_type_info = None

    def _store_dataset_info(self):
        """Store dataset information for later use."""
        self.num_nodes = int(self.full_data.num_nodes)
        self.node_features_dim = self.node_features.shape[1]
        self.edge_features_dim = self.edge_features.shape[1]

        if self.train_data is not None:
            print(f"  Training: {len(self.train_data.sources)} interactions")
        if self.val_data is not None:
            print(f"  Validation: {len(self.val_data.sources)} interactions")
        if self.test_data is not None:
            print(f"  Test: {len(self.test_data.sources)} interactions")
        print(f"  Nodes: {self.num_nodes}, Node feat dim: {self.node_features_dim}")

    def _create_pyg_data(
        self,
        temporal_data: TemporalData,
        node_type_info=None,
        stage="train",
        disjoint_message_label_edges=False,
    ):
        """Convert temporal data to PyTorch Geometric Data object."""
        if stage == "train":
            if disjoint_message_label_edges:
                # randomly mask 10% edges for link prediction labels
                msk = self.np_rng.random(len(temporal_data.sources)) < 0.1
                message_data = temporal_data.apply_mask(~msk)
                label_data = temporal_data.apply_mask(msk)
            else:
                message_data = temporal_data
                label_data = temporal_data
        elif stage == "val":
            val_timestamp = self.val_data.timestamps.min()
            msk = self.full_data.timestamps < val_timestamp
            message_data = self.train_data
            label_data = temporal_data
        elif stage == "test":
            test_timestamp = self.test_data.timestamps.min()
            msk = self.full_data.timestamps < test_timestamp
            message_data = self.full_data.apply_mask(msk)
            label_data = temporal_data

        # Convert to PyTorch tensors
        x = torch.from_numpy(self.node_features).float()
        edge_index = torch.stack(
            [
                torch.from_numpy(message_data.sources).long(),
                torch.from_numpy(message_data.destinations).long(),
            ],
            dim=0,
        )

        # Get edge features for these edges
        if hasattr(message_data, "edge_idxs") and message_data.edge_idxs is not None:
            edge_features_indices = message_data.edge_idxs
        else:
            edge_features_indices = np.arange(len(message_data.sources))
        edge_attr = torch.from_numpy(self.edge_features[edge_features_indices]).float()
        if hasattr(label_data, "edge_idxs") and label_data.edge_idxs is not None:
            edge_features_indices = label_data.edge_idxs
        else:
            edge_features_indices = np.arange(len(label_data.sources))
        edge_label_attr = torch.from_numpy(
            self.edge_features[edge_features_indices]
        ).float()

        # Labels and timestamps
        edge_label_index = torch.stack(
            [
                torch.from_numpy(label_data.sources).long(),
                torch.from_numpy(label_data.destinations).long(),
            ],
            dim=0,
        )
        edge_label = torch.ones_like(edge_label_index[0]).float()
        t = torch.from_numpy(message_data.timestamps).float()
        edge_label_t = torch.from_numpy(label_data.timestamps).float()

        # Negative sampling for edge labels
        if self.negative_sampling_ratio > 0:
            neg_edges = negative_sampling(
                edge_index=edge_label_index,
                num_nodes=int(self.num_nodes),
                num_neg_samples=self.negative_sampling_ratio,
                method="sparse",
            )
            edge_label_index = torch.cat([edge_label_index, neg_edges], dim=1)
            neg_edge_label = torch.zeros(neg_edges.size(1), dtype=torch.float)
            edge_label = torch.cat([edge_label, neg_edge_label], dim=0)
            edge_label_t = torch.cat(
                [edge_label_t, neg_edge_label.new_zeros(neg_edges.size(1))], dim=0
            )
            edge_label_attr = torch.cat(
                [
                    edge_label_attr,
                    edge_label_attr.new_zeros(
                        (neg_edges.size(1), edge_label_attr.size(1))
                    ),
                ],
                dim=0,
            )

        # Add node type information for inductive evaluation
        # Create masks for different edge types based on node information with bounds checking
        if self.mode == "inductive" and node_type_info is not None:
            is_new_node_mask = []
            edge_types = []
            for src, dst in edge_label_index.T.numpy():
                # Check bounds to avoid index errors
                src_is_new = (
                    src < len(node_type_info["is_new_node"])
                ) and node_type_info["is_new_node"][src]
                dst_is_new = (
                    dst < len(node_type_info["is_new_node"])
                ) and node_type_info["is_new_node"][dst]

                is_new_node_mask.append((src_is_new or dst_is_new).astype(int))

                # Determine edge type
                if src_is_new and dst_is_new:
                    edge_types.append(3)  # new-new
                elif src_is_new:
                    edge_types.append(1)  # new-old
                elif dst_is_new:
                    edge_types.append(2)  # old-new
                else:
                    edge_types.append(0)  # old-old

            is_new_node_mask = torch.from_numpy(np.array(is_new_node_mask, dtype=bool))
            edge_types = torch.tensor(edge_types)

            node_info = {"is_new_node": is_new_node_mask, "edge_types": edge_types}
        else:
            node_info = None

        data = Data(
            x=x,
            # edge data for message passing
            edge_index=edge_index,
            edge_attr=edge_attr,
            t=t,
            # edge data for supervision
            edge_label=edge_label,
            edge_label_index=edge_label_index,
            edge_label_attr=edge_label_attr,
            edge_label_t=edge_label_t,
            # data for inductive evaluation
            node_info=node_info,
        )

        return data

    def train_dataloader(self) -> DataLoader:
        """Return training data loader."""
        if self.train_data is None:
            raise ValueError("Training data not initialized. Call setup('fit') first.")

        # Convert to PyG Data object with node type information
        node_type_info = getattr(self, "train_node_type_info", None)
        data = self._create_pyg_data(self.train_data, node_type_info, "train", True)

        return DataLoader(
            [data],  # Single graph object
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle_train,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation data loader."""
        if self.val_data is None:
            raise ValueError(
                "Validation data not initialized. Call setup('fit') first."
            )

        # Convert to PyG Data object with node type information
        node_type_info = getattr(self, "val_node_type_info", None)
        self._val_pyg_data = self._create_pyg_data(
            self.val_data,
            node_type_info,
            stage="val",
        )

        return DataLoader(
            [self._val_pyg_data],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        if self.test_data is None:
            raise ValueError("Test data not initialized. Call setup('test') first.")

        # Convert to PyG Data object with node type information
        node_type_info = getattr(self, "test_node_type_info", None)
        self._test_pyg_data = self._create_pyg_data(
            self.test_data,
            node_type_info,
            stage="test",
        )

        return DataLoader(
            [self._test_pyg_data],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
        )

    def transfer_batch_to_device(
        self, batch, device: torch.device, dataloader_idx: int = 0
    ):
        """Transfer batch to device with custom handling for graph data."""
        # For graph data, transfer all tensors to device
        if isinstance(batch, list) and len(batch) == 1:
            batch = batch[0]  # Extract single Data object

        # Move all tensor attributes to device
        for attr in [
            "x",
            "edge_index",
            "edge_attr",
            "y",
            "t",
            "edge_label",
            "edge_label_index",
            "edge_label_t",
            "edge_label_attr",
            "n_id",
            "e_id",
        ]:
            if hasattr(batch, attr) and getattr(batch, attr) is not None:
                setattr(batch, attr, getattr(batch, attr).to(device))

        return batch
