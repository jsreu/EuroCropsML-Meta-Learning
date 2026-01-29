import json
import logging
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import matplotlib
import mlflow
from mlflow.exceptions import MlflowException

from eurocropsmeta.experiment.utils import configure_mlflow
from eurocropsmeta.models.base import Model
from eurocropsmeta.train.callback import TrainCallback
from eurocropsmeta.train.utils import ScalarMetric, TaskMetric

logger = logging.getLogger(__name__)


class MLFlowLogger:
    """Wrapped mlflow logger.

    Args:
        experiment_name: The name of the experiment.
    """

    def __init__(self, experiment_name: str):
        configure_mlflow(experiment_name)

    def start_run(self, run_name: str | None = None) -> None:
        """Start a new mflow run."""
        mlflow.start_run(run_name=run_name)

    def end_run(self, run_name: str | None = None) -> None:
        """End the current mflow run."""
        mlflow.end_run()

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags for the current run."""
        mlflow.set_tags(tags)

    def get_run_name(self) -> str:
        """Get the name of the current run."""
        return cast(str, mlflow.active_run().info.run_name)  # type: ignore[union-attr]

    @staticmethod
    def log(name: str, value: float | int, global_step: int | None = None) -> None:
        """Log scalar metrics.

        Args:
            name: The metric name
            value: The metric value
            global_step: The training step at which the value was obtained.
        """
        try:
            mlflow.log_metric(key=name, value=float(value), step=global_step)
        except MlflowException as e:
            logger.warning("mlflow exception: %s", str(e))

    @staticmethod
    def get_artifact_uri(artifact_path: str) -> str:
        """Get the full artifact uri for the current run and artifact path."""
        return mlflow.get_artifact_uri(artifact_path)

    @staticmethod
    def log_artifact(artifact_path: str) -> None:
        """Log model as artifacts."""
        try:
            mlflow.log_artifact(artifact_path)
        except MlflowException as e:
            logger.warning("mlflow exception: %s", str(e))

    @staticmethod
    def log_scalar_metrics(
        values_dict: dict[str, float | int], global_step: int | None = None
    ) -> None:
        """Log dictionary of metrics.

        Args:
            values_dict: Dictionary of metric names and values.
            global_step: The training step at which the values was obtained.
        """
        try:
            mlflow.log_metrics(values_dict, step=global_step)
        except MlflowException as e:
            logger.warning("mlflow exception: %s", str(e))

    @staticmethod
    def log_artifact_metrics(
        metrics: dict[str, TaskMetric], global_step: int | None = None
    ) -> None:
        """Log non-standard (tensor) metrics.

        Converts TaskMetric results into loggable artifacts and dispatches
        them to the respective logging method, based on artifact type.

        Args:
            metrics: Dictionary of metric names and TaskMetrics.
            global_step: The training step at which the values were obtained.

        """
        for name, metric in metrics.items():
            artifacts = metric.get_artifacts()
            if global_step is not None:
                name = f"{name}_{global_step:04d}"
            for artifact in artifacts:
                if isinstance(artifact, list):
                    MLFlowLogger.log_dict(artifact, f"{name}.json")
                elif isinstance(artifact, matplotlib.figure.Figure):
                    MLFlowLogger.log_figure(artifact, f"{name}.png")
                    matplotlib.pyplot.close(artifact)
                else:
                    logger.warning(
                        "unsupported artifact type %s for metric %s, skipped logging",
                        type(artifact),
                        name,
                    )

    @staticmethod
    def log_dict(dictionary: Any, name: str) -> None:
        """Log any JSON-serializable object, e.g. lists or dicts."""
        try:
            mlflow.log_dict(dictionary, name)
        except MlflowException as e:
            logger.warning("mlflow exception %s", str(e))

    @staticmethod
    def log_hparams(hparams: dict[str, str | int | float | None]) -> None:
        """Log model hyperparameters with performance metrics."""
        try:
            mlflow.log_params(hparams)
        except MlflowException as e:
            logger.warning("mlflow exception: %s", str(e))

    def log_json(self, name: str, data: dict[str, Any]) -> None:
        """Log json `data` to mlflow run under `name`."""
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir).joinpath(name + ".json")
            with open(path, "w") as f:
                json.dump(data, f)
            self.log_artifact(str(path.absolute()))

    @staticmethod
    def log_figure(fig: matplotlib.figure.Figure, name: str) -> None:
        """Log a Matplotlib figure."""
        try:
            mlflow.log_figure(fig, name)
        except MlflowException as e:
            logger.warning("mlflow exception %s", str(e))


class MLFlowCallback(TrainCallback):
    """Callback for logging to mlflow.

    Args:
        mlflow_logger: The mlflow wrapper to use for logging.
        log_model: If true, save model weights at end of training.
        log_model_on_validation: If true, save model weights at validation.
        log_artifact_metrics_on_validation: If true, log non-scalar metrics at validation.
    """

    def __init__(
        self,
        mlflow_logger: MLFlowLogger,
        log_model: bool = True,
        log_model_on_validation: bool = False,
        log_artifact_metrics_on_validation: bool = False,
    ):
        self.mlflow_logger = mlflow_logger
        self.log_model = log_model
        self.log_model_on_validation = log_model_on_validation
        self.log_artifact_metrics_on_validation = log_artifact_metrics_on_validation

    def start_callback(self, params: dict[str, Any]) -> None:
        """Function called at training start."""
        self.mlflow_logger.log_hparams(params)

    def train_callback(
        self,
        loss: float,
        metrics: Sequence[TaskMetric],
        model: Model,
        step: int | None = None,
    ) -> None:
        """Function called at training step."""
        scalar_metrics = {
            "train_" + metric.name: metric.get_scalar()
            for metric in metrics
            if isinstance(metric, ScalarMetric)
        }
        scalar_metrics["train_loss"] = loss
        self.mlflow_logger.log_scalar_metrics(scalar_metrics, global_step=step)

    def validation_callback(
        self,
        loss: float,
        metrics: Sequence[TaskMetric],
        model: Model,
        step: int | None = None,
    ) -> None:
        """Function called at validation step."""
        scalar_metrics: dict[str, float] = {}
        artifact_metrics: dict[str, TaskMetric] = {}
        for metric in metrics:
            if isinstance(metric, ScalarMetric):
                scalar_metrics["val_" + metric.name] = metric.get_scalar()
            else:
                artifact_metrics["val_" + metric.name] = metric
        scalar_metrics["val_loss"] = loss
        self.mlflow_logger.log_scalar_metrics(scalar_metrics, global_step=step)
        if self.log_model_on_validation:
            self._log_model(f"model_{step}", model)
        if self.log_artifact_metrics_on_validation:
            self.mlflow_logger.log_artifact_metrics(artifact_metrics, global_step=step)

    def test_callback(
        self,
        loss: float,
        metrics: Sequence[TaskMetric],
        model: Model,
        step: int | None = None,
    ) -> None:
        """Function called at test step."""
        scalar_metrics: dict[str, float] = {}
        artifact_metrics: dict[str, TaskMetric] = {}
        for metric in metrics:
            if isinstance(metric, ScalarMetric):
                scalar_metrics["test_" + metric.name] = metric.get_scalar()
            else:
                artifact_metrics["test_" + metric.name] = metric
        scalar_metrics["test_loss"] = loss
        self.mlflow_logger.log_scalar_metrics(scalar_metrics, global_step=step)
        self.mlflow_logger.log_artifact_metrics(artifact_metrics, global_step=step)

    def end_callback(self, model: Model) -> None:
        """Function called at end of training."""
        if self.log_model:
            self._log_model("model", model)

    def _log_model(self, name: str, model: Model) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir).joinpath("model")
            path.mkdir(parents=True)
            model.save(path)
            self.mlflow_logger.log_artifact(str(path.absolute()))
