from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MLP
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
    BinaryAveragePrecision,
)
from torchmetrics.retrieval import RetrievalMAP, RetrievalMRR
from torchmetrics import Metric, MetricCollection
from torch_geometric.typing import PairTensor

from tgx.utils.typing import ModelOutputs
from ..temporal_evaluator import TemporalEvaluator


class NodeClassificationEvaluator(TemporalEvaluator):
    """todo: Evaluator for node classification tasks."""

    require_node_embeddings = True

    def __init__(
        self,
        input_dim: int,
        num_classes: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        raise NotImplementedError("NodeClassificationEvaluator is not yet implemented.")
        # super().__init__(**kwargs)  # Don't store device

    #     self.num_classes = num_classes
    #     self.input_dim = input_dim
    #     self.hidden_dim = hidden_dim or input_dim  # Default to input_dim if not specified
    #     self.dropout = dropout

    #     # Initialize mlp_head immediately in __init__ if num_classes is known
    #     if num_classes is not None:
    #         channels = [input_dim, self.hidden_dim, num_classes]
    #         self.mlp_head = MLP(channels, dropout=dropout, act="relu")
    #     else:
    #         # For cases where num_classes might be determined later
    #         self.mlp_head = None

    # def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    #     """Compute cross entropy loss for node classification."""
    #     # Ensure tensors are on same device for CE loss
    #     if predictions.device != targets.device:
    #         targets = targets.to(predictions.device)
    #     else:
    #         targets = targets

    #     # Ensure targets are long for CE loss
    #     targets_long = targets.long()

    #     return F.cross_entropy(predictions, targets_long)

    # def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    #     """Compute node classification metrics."""
    #     # Get predicted classes
    #     if predictions.dim() > 1:
    #         pred_classes = predictions.argmax(dim=1)
    #     else:
    #         pred_classes = predictions

    #     pred_classes = pred_classes
    #     # Ensure tensors are on same device for comparison
    #     if pred_classes.device != targets.device:
    #         targets = targets.to(pred_classes.device)
    #     else:
    #         targets = targets
    #     true_classes = targets.long()

    #     # Calculate accuracy
    #     correct = (pred_classes == true_classes).sum().item()
    #     total = len(pred_classes)
    #     accuracy = correct / (total + 1e-10)

    #     # Calculate precision, recall, F1 per class
    #     num_classes = self.num_classes or (predictions.size(-1) if predictions.dim() > 1 else 2)

    #     precision_per_class = []
    #     recall_per_class = []
    #     f1_per_class = []

    #     for i in range(num_classes):
    #         true_mask = (true_classes == i)
    #         pred_mask = (pred_classes == i)

    #         tp = ((true_mask) & (pred_mask)).sum().item()
    #         fp = ((~true_mask) & (pred_mask)).sum().item()
    #         fn = ((true_mask) & (~pred_mask)).sum().item()
    #         tn = ((~true_mask) & (~pred_mask)).sum().item()

    #         precision = tp / (tp + fp + 1e-10)
    #         recall = tp / (tp + fn + 1e-10)
    #         f1 = 2 * precision * recall / (precision + recall + 1e-10)

    #         precision_per_class.append(precision)
    #         recall_per_class.append(recall)
    #         f1_per_class.append(f1)

    #     # Macro and micro averages
    #     macro_precision = sum(precision_per_class) / num_classes
    #     macro_recall = sum(recall_per_class) / num_classes
    #     macro_f1 = sum(f1_per_class) / num_classes

    #     return {
    #         'accuracy': accuracy,
    #         'macro_precision': macro_precision,
    #         'macro_recall': macro_recall,
    #         'macro_f1': macro_f1,
    #         'precision_per_class': precision_per_class,
    #         'recall_per_class': recall_per_class,
    #         'f1_per_class': f1_per_class,
    #         'num_classes': num_classes,
    #         'num_predictions': len(predictions),
    #         'num_correct': correct
    #     }

    # def log_metrics(self, metrics: Dict[str, float], stage: str, logger: Any) -> None:
    #     """Log node classification metrics."""
    #     # Log main metrics
    #     main_metrics = ['accuracy', 'macro_f1']
    #     for metric_name in main_metrics:
    #         if metric_name in metrics:
    #             logger.log_dict({
    #                 f'{stage}/{metric_name}': metrics[metric_name]
    #             })

    #     # Log main metric for optimization
    #     main_metric = 'accuracy'
    #     logger.log_dict({
    #         f'{stage}/main_metric': metrics.get(main_metric, 0.0)
    #     })

    #     # Log per-class metrics
    #     if 'precision_per_class' in metrics:
    #         for i, precision in enumerate(metrics['precision_per_class']):
    #             logger.log_dict({
    #                 f'{stage}/precision_class_{i}': precision
    #             })

    # def forward(self, model_output: torch.Tensor) -> torch.Tensor:
    #     """Forward pass through evaluator to get node classification logits."""
    #     if self.mlp_head is not None:
    #         predictions = self.mlp_head(model_output)
    #     else:
    #         predictions = model_output

    #     # For node classification, ensure proper shape [num_nodes, num_classes]
    #     return predictions

    # def post_process_predictions(self, predictions):
    #     emb = super().post_process_predictions(predictions)['node_embeddings']
    #     return self(emb)
