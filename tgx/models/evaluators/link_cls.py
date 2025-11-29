from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MLP

from tgx.utils.typing import ModelOutputs

from ..temporal_evaluator import TemporalEvaluator


class LinkClassificationEvaluator(TemporalEvaluator):
    """Evaluator for link classification tasks."""

    require_edge_embeddings = True

    def __init__(
        self,
        input_dim: int,
        num_classes: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)  # Don't store device
        self.num_classes = num_classes or 2  # Binary classification by default
        self.input_dim = input_dim
        self.hidden_dim = (
            hidden_dim or input_dim
        )  # Default to input_dim if not specified
        self.dropout = dropout
        self.label_smoothing = label_smoothing

        # Initialize mlp_head immediately in __init__ for link classification
        channels = [
            input_dim,
            self.hidden_dim,
            1 if self.num_classes == 2 else self.num_classes,
        ]
        self.mlp_head = MLP(channels, dropout=dropout, act="relu")

    def forward(self, edge_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through evaluator to get link classification probabilities."""
        predictions = self.mlp_head(edge_embeddings)

        # For link classification, ensure single output dimension for binary case
        if self.num_classes == 2 and predictions.dim() > 1:
            predictions = predictions.squeeze(-1)  # Remove last dimension for binary

        return predictions

    def compute_loss(
        self, outputs: ModelOutputs, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute binary cross entropy loss for link classification."""
        predictions = outputs["predictions"]
        # Ensure tensors are on same device for BCE loss
        if predictions.device != targets.device:
            targets = targets.to(predictions.device)

        if self.num_classes == 2:
            # Binary classification
            smoothed_targets = targets.float()
            smoothed_targets = smoothed_targets * (1.0 - self.label_smoothing) + (1.0 - smoothed_targets) * self.label_smoothing
            return F.binary_cross_entropy_with_logits(predictions, smoothed_targets)
        else:
            # Multi-class classification
            targets_long = targets.long()
            return F.cross_entropy(predictions, targets_long, label_smoothing=self.label_smoothing)

    def compute_metrics(
        self,
        outputs: ModelOutputs,
        targets: torch.Tensor,
        node_info: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Compute link classification metrics."""
        predictions = outputs["predictions"]
        if predictions.device != targets.device:
            targets = targets.to(predictions.device)

        if self.num_classes == 2:
            # Compute binary classification metrics
            metrics = self.metric_collection(predictions, targets)
            return {key: value.item() for key, value in metrics.items()}

    def log_metrics(self, metrics: Dict[str, float], stage: str, logger: Any) -> None:
        """Log link classification metrics."""
        for metric_name, metric_value in metrics.items():
            logger.log(f"{stage}/{metric_name}", metric_value)

        # Log main metric for optimization
        main_metric = "auroc"
        logger.log(f"{stage}/main_metric", metrics.get(main_metric, 0.0))

        # Log per-class metrics
        if "precision_per_class" in metrics:
            for i, precision in enumerate(metrics["precision_per_class"]):
                logger.log_dict({f"{stage}/precision_class_{i}": precision})

    def get_targets(self, batch:Data, outputs: ModelOutputs, stage) -> torch.LongTensor:
        return batch.edge_label.long().to(self.device)

    def postprocess_outputs(
        self,
        outputs: ModelOutputs,
        batch: Optional[Data] = None,
        stage: Optional[str] = None,
    ) -> ModelOutputs:
        outputs =  super().postprocess_outputs(outputs, batch=batch, stage=stage)
        predictions = self(outputs["edge_embeddings"])
        outputs["predictions"] = predictions
        return outputs