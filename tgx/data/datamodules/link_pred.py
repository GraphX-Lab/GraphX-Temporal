"""
DataModule for Link Prediction (LP) tasks.

This module provides specialized data loading and processing for link prediction
tasks using the benchtemp format with LPDatasetLoader.
"""

from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, LinkNeighborLoader

from tgx.dataset.lp import LPDatasetLoader
from ..temporal_data import Data as TemporalData
from .temporal_datamodule import TemporalDataModule


class LPDataModule(TemporalDataModule):
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
        num_neighbors: List[int] = [20, 20],
        mode: str = "transductive",
        historical_messages: bool = True,
        disjoint_training_edges: bool = False,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle_train=shuffle_train,
            split_ratio=split_ratio,
            mode=mode,
            num_neighbors=num_neighbors,
            **kwargs,
        )

        self.negative_sampling_ratio = negative_sampling_ratio
        self.different_new_nodes_between_val_and_test = (
            different_new_nodes_between_val_and_test
        )

        # LPDatasetLoader will be initialized in setup() to avoid circular import
        self.dataset_loader = None
        self.historical_messages = historical_messages
        self.disjoint_training_edges = disjoint_training_edges

        # Additional data containers specific to link prediction
        self.new_node_val_data = None
        self.new_node_test_data = None
        self.new_old_node_val_data = None
        self.new_old_node_test_data = None
        self.new_new_node_val_data = None
        self.new_new_node_test_data = None
        self.unseen_nodes = None
        self.unseen_nodes_num = None

        # Inductive node info
        self.train_node_type_info = None
        self.val_node_type_info = None
        self.test_node_type_info = None
        self.new_node_val_type_info = None
        self.new_node_test_type_info = None

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

        if self.disjoint_training_edges:
            self.training_disjoint_mask = (
                self.np_rng.random(len(self.train_data.sources)) < 0.1
            )
        else:
            self.training_disjoint_mask = None

        # Create node type information for inductive evaluation
        if self.mode == "inductive":
            self._create_node_type_info()

        # Store dataset information
        self._store_dataset_info(task_prefix="LP")

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

    def get_message_edges(
        self,
        stage,
        temporal_data=None,
    ):
        if stage == "train" and self.disjoint_training_edges:
            # For disjoint training, mask 10% edges for link prediction labels
            msk = self.training_disjoint_mask
            message_data = temporal_data.apply_mask(~msk)
            tensor_dict = message_data.to_tensor_dict()
            message_edges = {
                "edge_index": tensor_dict["edge_index"],
                "edge_attr": tensor_dict["edge_attr"],
                "edge_idxs": tensor_dict["edge_idx"],
                "edge_time": tensor_dict["timestamps"],
            }
        elif stage == "test" and self.mode == "inductive":
            # For inductive test, filter out edges with unseen nodes
            test_timestamp = self.test_data.timestamps.min()
            msk = np.logical_and.reduce(
                [
                    ~np.isin(self.full_data.sources, self.unseen_nodes),
                    ~np.isin(self.full_data.destinations, self.unseen_nodes),
                    self.full_data.timestamps < test_timestamp,
                ]
            )
            message_data = self.full_data.apply_mask(msk)
            tensor_dict = message_data.to_tensor_dict()
            message_edges = {
                "edge_index": tensor_dict["edge_index"],
                "edge_attr": tensor_dict["edge_attr"],
                "edge_idxs": tensor_dict["edge_idx"],
                "edge_time": tensor_dict["edge_time"],
            }
            tensor_dict = message_data.to_tensor_dict()
        else:
            message_edges = super().get_message_edges(stage, temporal_data)

        return message_edges

    def get_label_edges(self, stage, temporal_data=None):
        if stage == "train" and self.disjoint_training_edges:
            # For disjoint training, use the masked edges for link prediction labels
            msk = self.training_disjoint_mask
            label_data = temporal_data.apply_mask(msk)
            tensor_dict = label_data.to_tensor_dict()
            label_edges = {
                "edge_label_index": tensor_dict["edge_index"],
                "edge_label_idx": tensor_dict["edge_idx"],
                "edge_label": torch.ones_like(tensor_dict["edge_idx"]),
                "edge_label_time": tensor_dict["edge_time"],
                # "edge_label_attr": tensor_dict.get("edge_attr", None),    # not needed
            }

        else:
            label_edges = super().get_label_edges(stage, temporal_data)

        if "edge_label" in label_edges:
            # Negative sampling:  In case edge_label does not exist, it will be automatically created and represents a binary classification task (0 = negative edge, 1 = positive edge).
            del label_edges["edge_label"]
        return label_edges

    def add_inductive_node_info(self, pyg_data: Data, stage: str) -> Data:
        # Add node type information for inductive evaluation
        # Create masks for different edge types based on node information with bounds checking
        node_type_info = getattr(self, f"{stage}_node_type_info", None)
        if self.mode == "inductive" and node_type_info is not None:
            is_new_node_mask = []
            edge_types = []
            for src, dst in pyg_data.edge_label_index.T.numpy():
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
            edge_types = torch.tensor(edge_types, dtype=torch.int8)

            # node_info = {"is_new_node": is_new_node_mask, "edge_types": edge_types}
            pyg_data.is_new_node_edge = is_new_node_mask
            pyg_data.edge_types = edge_types
        else:
            # node_info = None
            pyg_data.is_new_node_edge = None
            pyg_data.edge_types = None
        return pyg_data

    def train_dataloader(self) -> DataLoader:
        """Return training data loader."""
        if self.train_data is None:
            raise ValueError("Training data not initialized. Call setup('fit') first.")

        # Convert to PyG Data object with node type information
        if self._train_pyg_data is None:
            self._train_pyg_data = self._create_pyg_data(self.train_data, "train")
        link_loader = LinkNeighborLoader(
            data=self._train_pyg_data,
            num_neighbors=self.num_neighbors,
            neg_sampling_ratio=self.negative_sampling_ratio,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            edge_label_index=self._train_pyg_data.edge_label_index,
            time_attr="edge_time",
            edge_label_time=self._train_pyg_data.edge_label_time,
            # edge_label=self._train_pyg_data.edge_label,
            transform=lambda batch: self.add_inductive_node_info(batch, stage="train"),
        )
        return link_loader

    def val_dataloader(self) -> DataLoader:
        """Return validation data loader."""
        if self.val_data is None:
            raise ValueError(
                "Validation data not initialized. Call setup('fit') first."
            )

        # Convert to PyG Data object with node type information
        if self._val_pyg_data is None:
            self._val_pyg_data = self._create_pyg_data(
                self.val_data,
                stage="val",
            )
        link_loader = LinkNeighborLoader(
            data=self._val_pyg_data,
            num_neighbors=self.num_neighbors,
            neg_sampling_ratio=self.negative_sampling_ratio,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            edge_label_index=self._val_pyg_data.edge_label_index,
            time_attr="edge_time",
            edge_label_time=self._val_pyg_data.edge_label_time,
            # edge_label=self._val_pyg_data.edge_label,
            transform=lambda batch: self.add_inductive_node_info(batch, stage="val"),
        )
        return link_loader

    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        if self.test_data is None:
            raise ValueError("Test data not initialized. Call setup('test') first.")

        if self._test_pyg_data is None:
            self._test_pyg_data = self._create_pyg_data(
                self.test_data,
                stage="test",
            )
        link_loader = LinkNeighborLoader(
            data=self._test_pyg_data,
            num_neighbors=self.num_neighbors,
            neg_sampling_ratio=self.negative_sampling_ratio,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            edge_label_index=self._test_pyg_data.edge_label_index,
            time_attr="edge_time",
            edge_label_time=self._test_pyg_data.edge_label_time,
            # edge_label=self._test_pyg_data.edge_label,
            transform=lambda batch: self.add_inductive_node_info(batch, stage="test"),
        )
        return link_loader
