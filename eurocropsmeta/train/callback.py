import logging
import math
from collections.abc import Sequence
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import optuna

from eurocropsmeta.models.base import Model
from eurocropsmeta.train.utils import ScalarMetric, TaskMetric

logger = logging.getLogger(__name__)


class TrainCallback:
    """Abstract class for training callbacks."""

    def start_callback(self, params: dict[str, Any]) -> None:
        """Function called at training start."""

    def train_callback(
        self,
        loss: float,
        metrics: Sequence[TaskMetric],
        model: Model,
        step: int | None = None,
    ) -> None:
        """Function called at training step."""

    def validation_callback(
        self,
        loss: float,
        metrics: Sequence[TaskMetric],
        model: Model,
        step: int | None = None,
    ) -> None:
        """Function called at validation step."""

    def test_callback(
        self,
        loss: float,
        metrics: Sequence[TaskMetric],
        model: Model,
        step: int | None = None,
    ) -> None:
        """Function called at test step."""

    def end_callback(self, model: Model) -> None:
        """Function called at end of training."""


class ValidationCallback(TrainCallback):
    """Callback to record (best) validation metrics."""

    def __init__(self, key_metric: str | dict[str, float]):
        self.key_metric = key_metric
        self.best_metric = math.inf if key_metric == "loss" else -math.inf
        self.best_metrics: dict[str, float | int] = {}
        self._checkpoint_path = Path(mkdtemp())

    def validation_callback(
        self,
        loss: float,
        metrics: Sequence[TaskMetric],
        model: Model,
        step: int | None = None,
    ) -> None:
        """Function called at validation step.

        It computes validation metrics, handles bookkeeping of
        the best `key_metric` value observed so far, and
        triggers model checkpoint saving whenever the
        `key_metric` improves.
        """
        if self.key_metric == "loss":
            current_metric = loss
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                self._save_checkpoint(model)
        else:
            scalar_metrics: dict[str, ScalarMetric] = {
                "val_" + metric.name: metric
                for metric in metrics
                if isinstance(metric, ScalarMetric)
            }
            if isinstance(self.key_metric, str):  # single key metric
                metric_name = "val_" + self.key_metric
                if metric_name not in scalar_metrics:
                    raise ValueError(f"Metric {metric_name} not found in logged scalar metrics.")
                current_metric = scalar_metrics[metric_name].get_scalar()
            else:  # multiple key metrics as weighted sum
                weighted_metric_values = [
                    scalar_metrics["val_" + metric_name].get_scalar() * metric_weight
                    for metric_name, metric_weight in self.key_metric.items()
                    if "val_" + metric_name in scalar_metrics
                ]
                if not weighted_metric_values:
                    raise ValueError("Weighted metrics not found in logged scalar metrics.")
                current_metric = sum(weighted_metric_values)

            # store all metrics current state if key metric was improved
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_metrics = {
                    name: metric.get_scalar() for name, metric in scalar_metrics.items()
                }
                self._save_checkpoint(model)

    def end_callback(self, model: Model) -> None:
        """Function called at end of training."""
        self._load_best_weights(model)
        self._clear()

    def _save_checkpoint(self, model: Model) -> None:
        """Saves a model checkpoint."""
        model.save(self._checkpoint_path)

    def _load_best_weights(self, model: Model) -> None:
        """Load best model weights."""
        if self._is_cleared:
            logger.warning(
                "Checkpoint directory is already cleared." "Can not load best weights. Skipped."
            )
        else:
            model.load(self._checkpoint_path, load_head=True)

    def _clear(self) -> None:
        """Clears the temporary checkpoint directory.

        `_load_best_weights()` can not be used anymore afterwards.
        """
        if self._is_cleared:
            logger.warning(
                "Checkpoint directory is already cleared." "Can not clear it again. Skipped."
            )
        else:
            logger.info("Clearing checkpoint directory.")
            for f in self._checkpoint_path.iterdir():
                f.unlink()
            self._checkpoint_path.rmdir()

    @property
    def _is_cleared(self) -> bool:
        """Check if `_clear()` has already been called."""
        return not self._checkpoint_path.exists()


class OptunaCallback(TrainCallback):
    """Callback to handle optuna tuning trial pruning."""

    def __init__(self, key_metric: str | dict[str, float], trial: optuna.Trial):
        self.key_metric = key_metric
        self.trial = trial

    def validation_callback(
        self,
        loss: float,
        metrics: Sequence[TaskMetric],
        model: Model,
        step: int | None = None,
    ) -> None:
        """Function called at validation step."""
        if self.key_metric == "loss":
            current_metric = loss
        else:
            scalar_metrics: dict[str, ScalarMetric] = {
                "val_" + metric.name: metric
                for metric in metrics
                if isinstance(metric, ScalarMetric)
            }
            if isinstance(self.key_metric, str):  # single key metric
                metric_name = "val_" + self.key_metric
                if metric_name not in scalar_metrics:
                    raise ValueError(f"Metric {metric_name} not found in logged scalar metrics.")
                current_metric = scalar_metrics[metric_name].get_scalar()
            else:  # multiple key metrics as weighted sum
                weighted_metric_values = [
                    scalar_metrics["val_" + metric_name].get_scalar() * metric_weight
                    for metric_name, metric_weight in self.key_metric.items()
                    if "val_" + metric_name in scalar_metrics
                ]
                if not weighted_metric_values:
                    raise ValueError("Weighted metrics not found in logged scalar metrics.")
                current_metric = sum(weighted_metric_values)
        if step is not None:
            self.trial.report(current_metric, step=step)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
