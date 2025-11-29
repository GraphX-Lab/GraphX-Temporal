from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MLP
from torchmetrics import MetricCollection

from tgx.utils.typing import ModelOutputs

from ..temporal_evaluator import TemporalEvaluator


class LinkPredictionEvaluator(TemporalEvaluator):
    """Evaluator for link prediction tasks with support for inductive evaluation."""

    require_edge_embeddings = True

    inductive_edge_types = ["new_old", "old_new", "new_new"]

    def __init__(
        self,
        input_dim: int,
        threshold: float = 0.5,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        output_dim: int = 1,
        mode: str = "transductive",  # "transductive" or "inductive",
        **kwargs,
    ):
        super().__init__(mode)  # Don't store device
        self.threshold = threshold
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = (
            hidden_dim or input_dim
        )  # Default to input_dim if not specified
        self.dropout = dropout
        self.mode = mode

        # Initialize mlp_head immediately in __init__ with device if provided
        channels = [input_dim, self.hidden_dim, output_dim]
        self.mlp_head = MLP(channels, dropout=dropout, act="relu")

        if self.mode == "inductive":
            # Inductive-specific metrics
            self.new_node_metrics = MetricCollection(
                {f"{n}/new": self.init_metric(n) for n in self.metric_names}
            )

            self.old_node_metrics = MetricCollection(
                {f"{n}/old": self.init_metric(n) for n in self.metric_names}
            )

            # Mixed edge type metrics (new-old, old-new, new-new)
            self.edge_type_metrics = MetricCollection(
                {
                    f"{n}/{m}": self.init_metric(n)
                    for n in self.metric_names
                    for m in self.inductive_edge_types
                }
            )

    def compute_loss(
        self, outputs: ModelOutputs, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute binary cross entropy loss for link prediction."""
        # Ensure targets are float for BCE loss
        targets_float = targets.float().to(self.device)
        predictions = outputs["predictions"]

        return F.binary_cross_entropy_with_logits(predictions, targets_float)

    def compute_metrics(
        self,
        outputs: ModelOutputs,
        targets: torch.Tensor,
        node_info: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Compute link prediction metrics with inductive support.

        Args:
            predictions: Model predictions (logits)
            targets: Ground truth labels
            node_info: Optional dict containing node type information for inductive evaluation
                     - 'is_new_node': Boolean tensor indicating edges with new nodes
                     - 'edge_types': Integer tensor indicating edge types (0=old-old, 1=new-old, 2=old-new, 3=new-new)
        """
        predictions = outputs["predictions"]
            
        # Convert logits to probabilities
        probs = torch.sigmoid(predictions)

        # Overall metrics (always computed)
        overall_metrics = self.metric_collection(probs, targets)
        result = {key: value.item() for key, value in overall_metrics.items()}

        # Inductive-specific metrics
        if node_info is not None:
            is_new_node = node_info.get(
                "is_new_node", torch.zeros_like(targets, dtype=bool)
            )
            edge_types = node_info.get(
                "edge_types", torch.zeros_like(targets, dtype=torch.long)
            )

            # Metrics for edges with new nodes
            if is_new_node.sum() > 0:
                new_node_metrics = self.new_node_metrics(
                    probs[is_new_node], targets[is_new_node]
                )
                result.update(
                    {key: value.item() for key, value in new_node_metrics.items()}
                )
            else:
                # If no new nodes, add zero metrics
                for metric_name in self.new_node_metrics.keys():
                    result[metric_name] = 0.0

            # Metrics for edges with only old nodes
            old_node_mask = ~is_new_node
            if old_node_mask.sum() > 0:
                old_node_metrics = self.old_node_metrics(
                    probs[old_node_mask], targets[old_node_mask]
                )
                result.update(
                    {key: value.item() for key, value in old_node_metrics.items()}
                )
            else:
                # If no old nodes, add zero metrics
                for metric_name in self.old_node_metrics.keys():
                    result[metric_name] = 0.0

            # Edge type specific metrics
            for edge_type, type_name in enumerate(self.inductive_edge_types):
                type_mask = (edge_types == edge_type)
                for metric_name in self.metric_names:
                    result_key = f"{metric_name}/{type_name}"
                    if type_mask.sum() > 0:
                        metric = self.edge_type_metrics[result_key](
                            probs[type_mask], targets[type_mask]
                        )
                        result[result_key] = metric.item()
                    else:
                        result[result_key] = 0.0

        return result

    def log_metrics(self, metrics: Dict[str, float], stage: str, logger: Any) -> None:
        """Log link prediction metrics."""
        for metric_name, metric_value in metrics.items():
            logger.log(f"{stage}/{metric_name}", metric_value)

        # For inductive, prioritize new node metrics
        main_metric = "auroc"

        logger.log(f"{stage}/main_metric", metrics.get(main_metric, 0.0))

    def forward(self, edge_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through evaluator to get link prediction probabilities."""
        predictions = self.mlp_head(edge_embeddings)

        # For link prediction, ensure single output dimension
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)

        return predictions

    def get_targets(self, batch:Data, outputs: ModelOutputs, stage) -> torch.LongTensor:
        return batch.edge_label.long().to(self.device) 

    def postprocess_outputs(
        self,
        outputs: ModelOutputs,
        batch: Optional[Data] = None,
        stage: Optional[str] = None,
    ) -> ModelOutputs:
        outputs =  super().postprocess_outputs(outputs, batch=batch, stage=stage)
        outputs["predictions"] = self(outputs["edge_embeddings"])
        return outputs


