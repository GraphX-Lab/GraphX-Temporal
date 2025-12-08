"""
Base DataModule for temporal graph learning tasks.

This module provides a common parent class for temporal graph datamodules,
sharing common initialization, data handling, and PyTorch Lightning integration.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, LinkNeighborLoader

from ..temporal_data import Data as TemporalData


class TemporalDataModule(pl.LightningDataModule, ABC):
    """
    Abstract base class for temporal graph data modules.

    This class provides common functionality for loading and processing temporal
    graph datasets across different tasks (link prediction, link classification, etc.).
    """

    def __init__(
        self,
        dataset: str = "mooc",
        data_dir: str = "./datasets",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle_train: bool = True,
        split_ratio: List[float] = [0.7, 0.15, 0.15],
        mode: str = "transductive",
        num_neighbors: List[int] = [20, 20],
        **kwargs,
    ):
        super().__init__()
        self.dataset = dataset
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.mode = mode
        self.num_neighbors = num_neighbors

        if len(split_ratio) == 2:
            split_ratio = [None] + split_ratio  # Train ratio will be inferred
        self.split_ratio = split_ratio

        # Initialize data containers as None (will be created in setup)
        self.node_features = None
        self.edge_features = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.full_data = None

        # Store basic dataset info
        self.num_nodes = None
        self.node_features_dim = None
        self.edge_features_dim = None

        # For caching PyG data objects
        self._train_pyg_data = None
        self._val_pyg_data = None
        self._test_pyg_data = None

        self.np_rng = np.random.default_rng(42)

    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training, validation, and testing.

        This method must be implemented by subclasses to handle the specific
        data loading logic for their task type.
        """
        pass

    @abstractmethod
    def _create_pyg_data(
        self, temporal_data: TemporalData, stage: str = "train", **kwargs
    ) -> Data:
        """
        Convert temporal data to PyTorch Geometric Data object.

        This method must be implemented by subclasses to handle the specific
        data conversion logic for their task type.
        """
        pass

    def _store_dataset_info(self, task_prefix: str = ""):
        """Store dataset information for later use."""
        if self.train_data is not None:
            self.num_nodes = (
                int(self.full_data.num_nodes)
                if self.full_data is not None
                else len(
                    set(self.train_data.sources).union(
                        set(self.train_data.destinations)
                    )
                )
            )
            self.node_features_dim = self.node_features.shape[1]
            self.edge_features_dim = self.edge_features.shape[1]

        print(f"{task_prefix} Dataset loaded: {self.dataset} ({self.mode} mode)")
        if self.train_data is not None:
            print(f"  Training: {len(self.train_data.sources)} interactions")
        if self.val_data is not None:
            print(f"  Validation: {len(self.val_data.sources)} interactions")
        if self.test_data is not None:
            print(f"  Test: {len(self.test_data.sources)} interactions")
        print(f"  Nodes: {self.num_nodes}, Node feat dim: {self.node_features_dim}")

    def train_dataloader(self) -> DataLoader:
        """Return training data loader."""
        if self.train_data is None:
            raise ValueError("Training data not initialized. Call setup('fit') first.")

        # Create PyG data if not cached
        if self._train_pyg_data is None:
            self._train_pyg_data = self._create_pyg_data(self.train_data, stage="train")

        link_loader = LinkNeighborLoader(
            data=self._train_pyg_data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            edge_label_index=self._train_pyg_data.edge_label_index,
            edge_label=self._train_pyg_data.edge_label,
            time_attr="edge_time",
            edge_label_time=self._train_pyg_data.edge_label_time,
        )
        return link_loader

    def val_dataloader(self) -> DataLoader:
        """Return validation data loader."""
        if self.val_data is None:
            raise ValueError(
                "Validation data not initialized. Call setup('fit') first."
            )

        # Create PyG data if not cached
        if self._val_pyg_data is None:
            self._val_pyg_data = self._create_pyg_data(self.val_data, stage="val")
        link_loader = LinkNeighborLoader(
            data=self._val_pyg_data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            edge_label_index=self._val_pyg_data.edge_label_index,
            edge_label=self._val_pyg_data.edge_label,
            time_attr="edge_time",
            edge_label_time=self._val_pyg_data.edge_label_time,
        )
        return link_loader

    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        if self.test_data is None:
            raise ValueError("Test data not initialized. Call setup('test') first.")

        # Create PyG data if not cached
        if self._test_pyg_data is None:
            self._test_pyg_data = self._create_pyg_data(self.test_data, stage="test")
        link_loader = LinkNeighborLoader(
            data=self._test_pyg_data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            edge_label_index=self._test_pyg_data.edge_label_index,
            edge_label=self._test_pyg_data.edge_label,
            time_attr="edge_time",
            edge_label_time=self._test_pyg_data.edge_label_time,
        )
        return link_loader

    def get_message_edges(
        self,
        stage: str,
        temporal_data: TemporalData = None,
    ):
        if temporal_data is None:
            temporal_data = getattr(self, f"{stage}_data")
        if stage == "train":
            message_data = temporal_data
        elif stage == "val":
            val_timestamp = temporal_data.timestamps.min()
            msk = self.full_data.timestamps < val_timestamp
            message_data = self.train_data
        elif stage == "test":
            test_timestamp = self.test_data.timestamps.min()
            msk = self.full_data.timestamps < test_timestamp
            message_data = self.full_data.apply_mask(msk)

        tensor_dict = message_data.to_tensor_dict()

        message_edges = {
            "edge_index": tensor_dict["edge_index"],
            "edge_attr": tensor_dict["edge_attr"],
            "edge_idxs": tensor_dict["edge_idx"],
            "edge_time": tensor_dict["edge_time"],
        }
        return message_edges

    def get_label_edges(
        self,
        stage: str,
        temporal_data: TemporalData = None,
    ):
        if temporal_data is None:
            temporal_data = getattr(self, f"{stage}_data")
        tensor_dict = temporal_data.to_tensor_dict()
        label_edges = {
            "edge_label_index": tensor_dict["edge_index"],
            "edge_label": tensor_dict["edge_labels"],
            "edge_label_time": tensor_dict["edge_time"],
            # 'edge_label_attr': tensor_dict.get('edge_attr', None),
        }
        return label_edges

    def _create_pyg_data(
        self, temporal_data: TemporalData, stage: str = "train"
    ) -> Data:
        """Convert temporal data to PyTorch Geometric Data object with node labels."""

        # Convert to PyTorch tensors
        x = torch.from_numpy(self.node_features).float()

        message_edges = self.get_message_edges(stage=stage, temporal_data=temporal_data)
        label_edges = self.get_label_edges(stage=stage, temporal_data=temporal_data)

        # For link classification, we use edge labels directly
        return Data(
            x=x,
            # data for message passing
            **message_edges,
            # data for link classification
            **label_edges,
        )
