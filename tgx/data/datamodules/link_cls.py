"""
DataModule for Link Classification (LC) tasks.

This module provides specialized data loading and processing for link classification
tasks using the benchtemp format with LCDataLoader.
"""

from typing import List, Optional


from tgx.dataset.lc import LCDatasetLoader
from ..temporal_data import Data as TemporalData
from .temporal_datamodule import TemporalDataModule


class LCDataModule(TemporalDataModule):
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
        num_neighbors: List[int] = [20, 20],
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

        self.use_validation = use_validation
        self.num_classes = num_classes

        # LCDataLoader will be initialized in setup() to avoid circular import
        self.lc_loader = None

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
                test_ratio=self.split_ratio[2],
            )

        # Load data using LCDataLoader
        full_data, node_features, edge_features, train_data, val_data, test_data = (
            self.lc_loader.load()
        )

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
        self._store_dataset_info(task_prefix="LC")
