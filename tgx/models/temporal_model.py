"""Base class for temporal graph models"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from lightning.pytorch import LightningModule
from torch_geometric.data import Data

from tgx.utils.typing import ModelOutputs

from .temporal_evaluator import DummyEvaluator, TemporalEvaluator


class TemporalModel(LightningModule, ABC):
    """
    Base class for temporal graph learning models with task-specific evaluation.

    This class extends PyTorch Lightning's LightningModule and provides
    a common interface for different temporal graph tasks with support
    for task-specific evaluators.

    Args:
        task (str): The specific task to perform (e.g., 'link_prediction', 'node_classification')
        learning_rate (float): Learning rate for optimization
        weight_decay (float): Weight decay for regularization
        evaluator_kwargs (dict): Additional arguments for the evaluator
        **kwargs: Additional model-specific parameters
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        evaluator: Optional[TemporalEvaluator] = DummyEvaluator(),
        **kwargs,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.evaluator = evaluator
        # if evaluator is not None:
        #     self.evaluator.model = self

        # Save hyperparameters for Lightning
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, batch: Data) -> ModelOutputs:
        pass

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Training step for Lightning."""
        # Get model predictions
        model_outputs = self(batch)

        # Apply task-specific output head and compute loss
        model_outputs = self.evaluator.postprocess_outputs(
            model_outputs, batch=batch, stage="train"
        )
        targets = self.evaluator.get_targets(batch, model_outputs, "train")
        loss = self.evaluator.compute_loss(model_outputs, targets)

        # Log training loss
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch.edge_label.size(-1),
        )

        # Compute and log task-specific metrics
        metrics = self.evaluator.compute_metrics(model_outputs, targets)
        self.evaluator.log_metrics(
            metrics, "train", self, batch_size=batch.edge_label.size(-1)
        )

        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Validation step for Lightning."""
        model_outputs = self(batch)
        model_outputs = self.evaluator.postprocess_outputs(
            model_outputs, batch=batch, stage="val"
        )
        targets = self.evaluator.get_targets(batch, model_outputs, "val")

        # Get node information for inductive evaluation
        if hasattr(batch, "is_new_node_edge"):
            node_info = {
                "is_new_node_edge": batch.is_new_node_edge,
                "edge_types": batch.edge_types,
            }
        else:
            node_info = None

        loss = self.evaluator.compute_loss(model_outputs, targets)

        # Log validation loss
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.edge_label.size(-1),
        )

        # Compute and log task-specific metrics
        metrics = self.evaluator.compute_metrics(model_outputs, targets, node_info)

        self.evaluator.log_metrics(
            metrics, "val", self, batch_size=batch.edge_label.size(-1)
        )

        return loss

    def test_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Test step for Lightning."""
        model_outputs = self(batch)
        model_outputs = self.evaluator.postprocess_outputs(
            model_outputs, batch=batch, stage="test"
        )
        targets = self.evaluator.get_targets(batch, model_outputs, "test")

        # Get node information for inductive evaluation
        if hasattr(batch, "is_new_node_edge"):
            node_info = {
                "is_new_node_edge": batch.is_new_node_edge,
                "edge_types": batch.edge_types,
            }
        else:
            node_info = None

        loss = self.evaluator.compute_loss(model_outputs, targets)

        # Log test loss
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.edge_label.size(-1),
        )

        # Compute and log task-specific metrics
        metrics = self.evaluator.compute_metrics(model_outputs, targets, node_info)

        self.evaluator.log_metrics(
            metrics, "test", self, batch_size=batch.edge_label.size(-1)
        )

        return loss

    def configure_optimizers(self):
        """Configure optimizers for Lightning."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer

    def on_train_start(self):
        # if self.trainer.logger is not None:
        class_name = self.__class__.__name__
        name = getattr(self, "model_name", class_name)
        # self.save_hyperparameters({'Model': name})
        if self.trainer.logger is not None:
            self.trainer.logger.log_hyperparams({"Model": name})
        return super().on_train_start()


