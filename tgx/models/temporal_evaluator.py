"""
Universal evaluator for different downstream tasks in temporal graph learning.

This module provides task-specific evaluation metrics and logging functions
that can be used across different models and tasks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.typing import PairTensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from torchmetrics.retrieval import RetrievalMAP, RetrievalMRR

from tgx.utils.typing import ModelOutputs

METRIC_MAPPING = {
    "acc": BinaryAccuracy,
    "precision": BinaryPrecision,
    "recall": BinaryRecall,
    "f1": BinaryF1Score,
    "auroc": BinaryAUROC,
    "ap": BinaryAveragePrecision,
    "map": RetrievalMAP,
    "mrr": RetrievalMRR,
}


class TemporalEvaluator(nn.Module, ABC):
    """
    Abstract base class for temporal graph task evaluators.

    Provides common interface for different downstream tasks:
    - Link Prediction (LP)
    - Node Classification (NC)
    """
    model = None
    require_node_embeddings = False
    require_edge_embeddings = False

    def __init__(
        self, mode: Optional[str] = None, metric_names: Optional[List[str]] = None
    ):
        super().__init__()
        self.mode = mode
        self.metric_names = ["auroc", "ap"] if metric_names is None else metric_names

        # Overall metrics (always computed)
        self.metric_collection = MetricCollection(
            {n: self.init_metric(n) for n in self.metric_names}
        )

    @abstractmethod
    def compute_loss(
        self, outputs: ModelOutputs, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute task-specific loss."""
        pass

    @abstractmethod
    def compute_metrics(
        self,
        outputs: ModelOutputs,
        targets: torch.Tensor,
        node_info: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Compute task-specific metrics."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], stage: str, logger: Any) -> None:
        """Log metrics for Lightning."""
        pass

    @abstractmethod
    def forward(self, model_output: torch.Tensor) -> torch.Tensor:
        """Forward pass through evaluator to get task-specific predictions."""
        pass

    def preprocess_inputs(self, batch: Data, stage: str) -> Data:
        """
        Preprocess inputs before passing to evaluator.
        Args:
            batch: Input batch
            stage: Training stage (train/val/test)
        return batch
        """
        return batch

    def get_targets(self, batch: Data, outputs: ModelOutputs,stage: str) -> torch.Tensor:
        """
        Extract targets based on the current task.

        Args:
            batch: Input batch
            stage: Training stage (train/val/test)

        Returns:
            Target tensor
        """
        return batch.y

    def postprocess_outputs(
        self,
        model_outputs: ModelOutputs,
        batch: Optional[Data] = None,
        stage: Optional[str] = None,
    ) -> ModelOutputs:
        """Optional post-processing of model_outputs (e.g., thresholding)."""
        if self.require_edge_embeddings:
            if "edge_embeddings" not in model_outputs:
                try:
                    edge_embeddings = self.create_edge_embeddings(
                        model_outputs["node_embeddings"],
                        batch["edge_label_index"],
                        edge_attr=model_outputs.get("edge_attr", None),
                    )
                    model_outputs["edge_embeddings"] = edge_embeddings
                except KeyError:
                    raise ValueError(
                        "Edge embeddings required but not found in model returned predictions, and could not be created from node embeddings and edge index."
                    )

        return model_outputs

    def create_edge_embeddings(
        self,
        node_embeddings: torch.Tensor | PairTensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Create edge embeddings by summing source and target node embeddings."""
        if isinstance(node_embeddings, torch.Tensor):
            source_embeddings = node_embeddings[edge_index[0]]
            target_embeddings = node_embeddings[edge_index[1]]
        else:
            source_embeddings, target_embeddings = node_embeddings
        edge_embeddings = (
            source_embeddings + target_embeddings
        )  # Simple sum; can be modified
        return edge_embeddings

    def init_metric(self, metric_name: str, **kwargs) -> Metric:
        cls = METRIC_MAPPING.get(metric_name, None)
        if cls is None:
            raise ValueError(f"Metric '{metric_name}' is not recognized.")
        return cls(**kwargs)

    def __repr__(self):
        # Task-specific evaluator representation
        return f"{self.__class__.__name__}(mode={self.mode})"

    @property
    def device(self) -> torch.device:
        """Get the device of the evaluator's parameters."""
        return next(self.parameters()).device


class DummyEvaluator(TemporalEvaluator):
    """A dummy evaluator that performs no evaluation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(
        self, outputs: ModelOutputs, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return zero loss."""
        return torch.tensor(0.0, device=self.device)

    def compute_metrics(
        self,
        outputs: ModelOutputs,
        targets: torch.Tensor,
        node_info: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Return empty metrics."""
        return {}

    def log_metrics(self, metrics: Dict[str, float], stage: str, logger: Any) -> None:
        """No-op for logging."""
        pass

    def forward(self, model_output: torch.Tensor) -> torch.Tensor:
        """Return model output as-is."""
        return model_output

