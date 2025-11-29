"""
DataModule for Link Classification (LC) tasks.

This module provides specialized data loading and processing for link classification
tasks using the benchtemp format with LCDataLoader.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tgx.dataset.lc import LCDatasetLoader
from ..temporal_data import Data as TemporalData

class LCDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for link classification tasks.

    This module handles loading and preparation of temporal graph datasets
    specifically for link classification with proper train/val/test splitting
    and edge-level processing using LCDataLoader.

    Args:
        dataset (str): Dataset name ('mooc', 'reddit', 'wikipedia', etc.)
        data_dir (str): Directory containing data files
        batch_size (int): Batch size for training
        num_workers (int): Number of data loading workers
        pin_memory (bool): Whether to pin memory for faster GPU transfer
        shuffle_train (bool): Whether to shuffle training data
        mode (str): 'transductive' or 'inductive' mode
        use_validation (bool): Whether to use validation split
        num_classes (Optional[int]): Number of classes (auto-detect if None)
    """

    def __init__(
        self,
        dataset: str = "mooc",
        data_dir: str = "./datasets",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle_train: bool = True,
        mode: str = "transductive",
        use_validation: bool = True,
        num_classes: Optional[int] = None,
        split_ratio: List[float] = [0.7, 0.15, 0.15],
        **kwargs
    ):
        super().__init__()
        self.dataset = dataset
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.mode = mode
        self.use_validation = use_validation
        self.num_classes = num_classes
        if len(split_ratio) == 2:
            split_ratio = [None] + split_ratio  # Train ratio will be inferred
        self.split_ratio = split_ratio

        # LCDataLoader will be initialized in setup() to avoid circular import
        self.lc_loader = None

        # Initialize data containers as None (will be created in setup)
        self.node_features = None
        self.edge_features = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

        # Store basic dataset info
        self.num_nodes = None
        self.node_features_dim = None
        self.edge_features_dim = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing using LCDataLoader."""

        # Initialize LCDataLoader if not already done
        if self.lc_loader is None:
            self.lc_loader = LCDatasetLoader(
                dataset_path=str(self.data_dir) + "/",
                dataset_name=self.dataset,
                use_validation=self.use_validation,
                train_ratio=self.split_ratio[0],
                val_ratio=self.split_ratio[1],
                test_ratio=self.split_ratio[2]
            )

        # Load data using LCDataLoader
        full_data, node_features, edge_features, train_data, val_data, test_data = self.lc_loader.load()

        # Store loaded data
        self.full_data: TemporalData = full_data
        self.node_features = node_features
        self.edge_features = edge_features
        self.train_data: TemporalData = train_data
        self.val_data: TemporalData = val_data
        self.test_data: TemporalData = test_data

        # For link classification, we use binary classification (0/1)
        if self.num_classes is None:
            self.num_classes = 2  # Binary link classification

        # Store dataset information
        self._store_dataset_info()

    def _store_dataset_info(self):
        """Store dataset information for later use."""
        if self.train_data is not None:
            self.num_nodes = len(set(self.train_data.sources).union(set(self.train_data.destinations)))
            self.node_features_dim = self.node_features.shape[1]
            self.edge_features_dim = self.edge_features.shape[1]

        print(f"LC Dataset loaded: {self.dataset} ({self.mode} mode)")
        if self.train_data is not None:
            print(f"  Training: {len(self.train_data.sources)} interactions")
        if self.val_data is not None:
            print(f"  Validation: {len(self.val_data.sources)} interactions")
        if self.test_data is not None:
            print(f"  Test: {len(self.test_data.sources)} interactions")
        print(f"  Nodes: {self.num_nodes}, Node feat dim: {self.node_features_dim}, Classes: {self.num_classes}")

    def _create_pyg_data(self, temporal_data: TemporalData, stage: str = 'train') -> Data:
        """Convert temporal data to PyTorch Geometric Data object with node labels."""

        if stage == 'train':
            message_data = temporal_data
        elif stage == 'val':
            val_timestamp = self.val_data.timestamps.min()
            msk = self.full_data.timestamps < val_timestamp
            message_data = self.train_data
        elif stage == 'test':
            test_timestamp = self.test_data.timestamps.min()
            msk = self.full_data.timestamps < test_timestamp
            message_data = self.full_data.apply_mask(msk)

        # Convert to PyTorch tensors
        x = torch.from_numpy(self.node_features).float()

        edge_index = torch.stack([
            torch.from_numpy(message_data.sources).long(),
            torch.from_numpy(message_data.destinations).long()
        ], dim=0)
        t = torch.from_numpy(message_data.timestamps).float()

        # Get edge features for these edges
        if hasattr(message_data, 'edge_idxs') and message_data.edge_idxs is not None:
            edge_features_indices = message_data.edge_idxs
        else:
            edge_features_indices = np.arange(len(message_data.sources))
        edge_attr = torch.from_numpy(self.edge_features[edge_features_indices]).float()
        if hasattr(temporal_data, 'edge_idxs') and temporal_data.edge_idxs is not None:
            edge_features_indices = temporal_data.edge_idxs
        else:
            edge_features_indices = np.arange(len(temporal_data.sources))
        edge_label_attr = torch.from_numpy(self.edge_features[edge_features_indices]).float()

        # Edge labels and timestamps
        edge_label = torch.from_numpy(temporal_data.labels).float()
        edge_label_index = torch.stack([
            torch.from_numpy(temporal_data.sources).long(),
            torch.from_numpy(temporal_data.destinations).long()
        ], dim=0)
        edge_label_t = torch.from_numpy(temporal_data.timestamps).float()

        # For link classification, we use edge labels directly
        return Data(
            x=x,
            # data for message passing
            edge_index=edge_index,
            edge_attr=edge_attr,
            t=t,
            # data for link classification
            edge_label=edge_label,  # Edge labels (0/1 for binary classification)
            edge_label_index=edge_label_index,  # Indices of edges with labels
            edge_label_attr=edge_label_attr,
            edge_label_t=edge_label_t  # Timestamps of labeled edges
        )

    def train_dataloader(self) -> DataLoader:
        """Return training data loader."""
        if self.train_data is None:
            raise ValueError("Training data not initialized. Call setup('fit') first.")

        # Convert to PyG Data object
        data = self._create_pyg_data(self.train_data)

        return DataLoader(
            [data],  # Single graph object
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle_train,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation data loader."""
        if self.val_data is None:
            raise ValueError("Validation data not initialized. Call setup('fit') first.")

        data = self._create_pyg_data(self.val_data, stage="val")

        return DataLoader(
            [data],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        if self.test_data is None:
            raise ValueError("Test data not initialized. Call setup('test') first.")

        data = self._create_pyg_data(self.test_data, stage="test")

        return DataLoader(
            [data],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=self.num_workers > 0
        )

    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int = 0):
        """Transfer batch to device with custom handling for graph data."""
        # For graph data, transfer all tensors to device
        if isinstance(batch, list) and len(batch) == 1:
            batch = batch[0]  # Extract single Data object

        # Move all tensor attributes to device
        for attr in ['x', 'edge_index', 'edge_attr', 'y', 't', 'edge_label', 'edge_label_index', 'edge_label_attr', 'edge_label_t']:
            if hasattr(batch, attr) and getattr(batch, attr) is not None:
                setattr(batch, attr, getattr(batch, attr).to(device))

        return batch