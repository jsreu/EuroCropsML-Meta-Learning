from __future__ import annotations

import logging
import random
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
import torchmetrics.classification

from eurocropsmeta.models.base import Model

logger = logging.getLogger(__name__)


class TaskMetric:
    """Light wrapper around torchmetric.Metric with custom logging representations.

    Args:
        name: Name of the metric.
        metric: A torchmetric.Metric instance to wrap.

    """

    def __init__(self, name: str, metric: torchmetrics.Metric) -> None:
        self.name = name
        self.metric = metric

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Any:
        """Update the internal state of the metric and return value on inputs."""
        return self.metric(pred, target)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update the internal state of the metric."""
        self.metric.update(pred, target)

    def compute(self) -> Any:
        """Compute the final metric value from internal state."""
        return self.metric.compute()

    def reset(self) -> None:
        """Reset the internal state of the matric."""
        self.metric.reset()

    @abstractmethod
    def get_artifacts(self) -> list[Any]:
        """Get the metric state as a list loggable artifacts."""

    def __repr__(self) -> str:
        """Text representation of the metric."""
        return f"{self.__class__.__name__}({self.name})"


class ScalarMetric(TaskMetric):
    """Light wrapper around torchmetric.Metrics that yield scalar values."""

    @classmethod
    def from_name(
        cls, metric_name: str, name: str | None = None, *args: Any, **kwargs: Any
    ) -> ScalarMetric:
        """Create ScalarMetric from the name of a torchmetrics.Metrics.

        Args:
            metric_name: Name of the torchmetrics.Metric to wrap.
            name: Optional custom name for the wrappend metric.
                If not specified, the torchmetric.Metric name will be used.
            *args: Other arguments passed on to the torchmetrics.Metric constructor.
            **kwargs: Other keyword arguments passed on to the torchmetrics.Metric constructor.

        Returns:
            The created ScalarMetric instance wrapping a torchmetric.Metric.
        """
        metric = ScalarMetric._get_scalar_metric_from_name(metric_name, *args, **kwargs)
        return ScalarMetric(name or metric_name, metric)

    def get_scalar(self) -> float:
        """Get the metric state represented as a scalar value."""
        return self.metric.compute().item()  # type: ignore[no-any-return]

    def get_artifacts(self) -> list[float]:
        """Get the metric state as a list of loggable artifacts.

        For a ScalarMetric the available artifact type is a scalar value.
        """
        return [self.get_scalar()]

    def __add__(
        self, other: ScalarMetric | torchmetrics.Metric | int | float | torch.Tensor
    ) -> ScalarMetric:
        """(Left) Addition of ScalarMetrics. Name will be copied from self."""
        if isinstance(other, ScalarMetric):
            other = other.metric
        metric = torchmetrics.metric.CompositionalMetric(torch.add, self.metric, other)
        return ScalarMetric(self.name, metric)

    def __radd__(self, other: torchmetrics.Metric | int | float) -> ScalarMetric:
        """(Right) addition of ScalarMetrics. Name will be copied from self."""
        metric = torchmetrics.metric.CompositionalMetric(torch.add, other, self.metric)
        return ScalarMetric(self.name, metric)

    def __sub__(
        self, other: ScalarMetric | torchmetrics.Metric | int | float | torch.Tensor
    ) -> ScalarMetric:
        """(Left) subtraction of ScalarMetrics. Name will be copied from self."""
        if isinstance(other, ScalarMetric):
            other = other.metric
        metric = torchmetrics.metric.CompositionalMetric(torch.sub, self.metric, other)
        return ScalarMetric(self.name, metric)

    def __rsub__(self, other: torchmetrics.Metric | int | float) -> ScalarMetric:
        """(Right) subtraction of ScalarMetrics. Name will be copied from self."""
        metric = torchmetrics.metric.CompositionalMetric(torch.sub, other, self.metric)
        return ScalarMetric(self.name, metric)

    def __mul__(
        self, other: ScalarMetric | torchmetrics.Metric | int | float | torch.Tensor
    ) -> ScalarMetric:
        """(Left) multiplication of ScalarMetrics. Name will be copied from self."""
        if isinstance(other, ScalarMetric):
            other = other.metric
        metric = torchmetrics.metric.CompositionalMetric(torch.mul, self.metric, other)
        return ScalarMetric(self.name, metric)

    def __rmul__(self, other: torchmetrics.Metric | int | float) -> ScalarMetric:
        """(Right) subtraction of ScalarMetrics. Name will be copied from self."""
        metric = torchmetrics.metric.CompositionalMetric(torch.mul, other, self.metric)
        return ScalarMetric(self.name, metric)

    def __truediv__(
        self, other: ScalarMetric | torchmetrics.Metric | int | float | torch.Tensor
    ) -> ScalarMetric:
        """(Left) division of ScalarMetrics. Name will be copied from self."""
        if isinstance(other, ScalarMetric):
            other = other.metric
        metric = torchmetrics.metric.CompositionalMetric(torch.true_divide, self.metric, other)
        return ScalarMetric(self.name, metric)

    def __rtruediv__(self, other: torchmetrics.Metric | int | float) -> ScalarMetric:
        """(Right) subtraction of ScalarMetrics. Name will be copied from self."""
        metric = torchmetrics.metric.CompositionalMetric(torch.true_divide, other, self.metric)
        return ScalarMetric(self.name, metric)

    @staticmethod
    def _get_scalar_metric_from_name(
        metric_name: str, *args: Any, **kwargs: Any
    ) -> torchmetrics.Metric:
        try:
            metric: torchmetrics.Metric = getattr(torchmetrics.classification, metric_name)(
                *args, **kwargs
            )
        except Exception as e:
            raise ValueError(
                f"Could not instantiate a torchmetric.Metric with name {metric_name}. {e}"
            ) from e
        return metric


class ConfusionMatrix(TaskMetric):
    """TaskMetric that stores and computes a classification confusion matrix.

    Args:
        class_names: The names of classes to be used for logging.
            If not specified, class names will be numbered 0,...,num_classes.
        log_json: Flag to turn on or off logging the metric as a JSON artifact.
        log_figure: Flag to turn on or off logging the metric as an image/figure.
        show_relatives: Mode for showing relative frequencies in the figure artifact.
            Ignored if log_figure is False. Options are "targets" (normalize per row),
            "predictions" (normalize per column), and "total" (normalize by rows and columns).
    """

    def __init__(
        self,
        class_names: Sequence[str] | None = None,
        log_json: bool = True,
        log_figure: bool = True,
        show_relatives: Literal["targets", "predictions", "total"] = "targets",
        *args: Any,
        **kwargs: Any,
    ):
        metric = torchmetrics.classification.ConfusionMatrix(*args, **kwargs)
        if log_json and log_figure:
            name = "ConfusionMatrixFigureAndJSON"
        elif log_json:
            name = "ConfusionMatrixJSON"
        elif log_figure:
            name = "ConfusionMatrixFigure"
        else:
            raise ValueError(
                "At least one of the following logging artifacts must be used: JSON, Figure"
            )
        super(ConfusionMatrix, self).__init__(name, metric)
        self.log_json = log_json
        self.log_figure = log_figure
        self.class_names = class_names
        self.show_relatives = show_relatives

    def _get_json_artifact(self, computed: torch.Tensor) -> list:
        """Convert matrix to loggable JSON artifact."""
        return computed.tolist()

    def _get_figure_artifact(
        self,
        computed: torch.Tensor,
    ) -> matplotlib.figure.Figure:
        """Convert matrix to loggable Figure artifact."""

        def _is_light_color(color: tuple[float, ...]) -> bool:
            """Auxiliary check to decide if an (r,g,b) color is perceived light or dark."""
            r, g, b = color[:3]  # trim to rgb in case it was rgba
            r_fac, g_fac, b_fac = (
                0.299,
                0.587,
                0.114,
            )  # rgb perception factors as per ITU-BT-601
            threshold = 0.584  # light vs dark perception cutoff
            luma = r_fac * r + g_fac * g + b_fac * b
            return luma > threshold

        assert computed.shape[0] == computed.shape[1]  # should be square, i.e. height=width
        num = computed.shape[0]
        if self.class_names is None:
            fig_class_names = list(map(str, range(num)))
        else:
            fig_class_names = list(self.class_names)
        assert len(fig_class_names) == num

        # extend confusion matrix by one summary row and column per class
        computed = torch.cat((computed, computed.sum(dim=0, keepdim=True)), dim=0)
        computed = torch.cat((computed, computed.sum(dim=1, keepdim=True)), dim=1)

        # create and format figure
        normalizer = matplotlib.colors.LogNorm(vmin=1, vmax=computed[-1, -1].item(), clip=True)
        cmap = plt.get_cmap("Blues")
        cm_to_inch = 1 / 2.54
        cm_per_cell = 1
        fsize = (num + 7) * cm_per_cell * cm_to_inch  # adjust figure size to num classes
        fig, ax = plt.subplots(figsize=(fsize, fsize))
        ax.matshow(computed, cmap=cmap, norm=normalizer)
        for idx, val in np.ndenumerate(computed):
            textcolor = "black" if _is_light_color(cmap(normalizer(val))) else "white"
            if self.show_relatives == "targets":
                rel_val = val / computed[idx[0], -1]
            elif self.show_relatives == "predictions":
                rel_val = val / computed[-1, idx[1]]
            elif self.show_relatives == "total":
                rel_val = val / computed[-1, -1]
            else:
                raise ValueError(
                    f"Invalid mode {self.show_relatives} for showing relative frequencies."
                )
            ax.text(
                x=idx[1],
                y=idx[0],
                s="{:d}\n{:1.2f}%".format(val, 100 * rel_val),
                va="center",
                ha="center",
                color=textcolor,
                fontsize="x-small",
            )

        ax.set_xticks(list(range(num + 1)))
        ax.set_xticklabels(fig_class_names + ["sum"], rotation=-30, ha="right", fontsize="xx-small")
        ax.set_yticks(list(range(num + 1)))
        ax.set_yticklabels(
            fig_class_names + ["sum"], rotation=-30, va="bottom", fontsize="xx-small"
        )
        ax.tick_params(
            direction="out",
            left=True,
            bottom=False,
            right=False,
            top=True,
            labelleft=True,
            labelbottom=False,
            labelright=False,
            labeltop=True,
        )

        ax.xaxis.set_label_position("top")
        ax.yaxis.set_label_position("left")
        ax.set_xlabel("Predictions")
        ax.set_ylabel("Targets")
        ax.set_title("Confusion Matrix")

        rect_main = matplotlib.patches.Rectangle(
            (-0.5, -0.5), num, num, fill=False, edgecolor="black", linewidth=3
        )
        rect_row_sum = matplotlib.patches.Rectangle(
            (num - 0.5, -0.5), 1.0, num, fill=False, edgecolor="black", linewidth=3
        )
        rect_col_sum = matplotlib.patches.Rectangle(
            (-0.5, num - 0.5), num, 1.0, fill=False, edgecolor="black", linewidth=3
        )
        rect_total_sum = matplotlib.patches.Rectangle(
            (num - 0.5, num - 0.5),
            1.0,
            1.0,
            fill=False,
            edgecolor="black",
            linewidth=3,
        )
        ax.add_patch(rect_main)
        ax.add_patch(rect_row_sum)
        ax.add_patch(rect_col_sum)
        ax.add_patch(rect_total_sum)

        fig.tight_layout()

        return fig

    def get_artifacts(self) -> list[Any]:
        """Get the metric state as a list of loggable artifacts.

        For a ConfusionMatrix the available artifact types are
        JSON and Figure (see __init__).
        """
        computed = self.metric.compute()
        artifacts: list[Any] = []
        if self.log_json:
            artifacts.append(self._get_json_artifact(computed))
        if self.log_figure:
            artifacts.append(self._get_figure_artifact(computed))

        return artifacts


def set_seed(seed: int) -> None:
    """Set random seed across python, torch and numpy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_metrics(
    metrics_list: Sequence[str],
    num_classes: int,
    class_names: Sequence[str] | None = None,
) -> Sequence[TaskMetric]:
    """Function to assign metrics to evaluate during training/testing.

    Args:
        metrics_list: List of metrics to return.
        num_classes: Number of classes.
        class_names: Optional list of class names. Metrics may use it for visualization purposes.

    Returns:
        Dictionary of metrics
    """
    task = "multiclass" if num_classes > 2 else "binary"
    default_metrics: Sequence[TaskMetric] = [
        # ScalarMetric.from_name(
        #     metric_name="MeanSquaredError",
        #     name="MSE",
        # ),
        ScalarMetric.from_name(
            metric_name="Accuracy",
            name="Acc",
            task=task,
            num_classes=num_classes,
        ),
        ScalarMetric.from_name(
            metric_name="F1Score",
            name="F1Score_macro",
            task=task,
            average="macro",
            num_classes=num_classes,
        ),
        ScalarMetric.from_name(
            metric_name="F1Score",
            name="F1Score_micro",
            task=task,
            average="micro",
            num_classes=num_classes,
        ),
        ScalarMetric.from_name(
            metric_name="CohenKappa",
            task=task,
            num_classes=num_classes,
        ),
        ConfusionMatrix(
            class_names=class_names,
            log_json=False,
            log_figure=True,
            task=task,
            num_classes=num_classes,
        ),
        ConfusionMatrix(
            class_names=class_names,
            log_json=True,
            log_figure=False,
            task=task,
            num_classes=num_classes,
        ),
        ConfusionMatrix(
            class_names=class_names,
            log_json=True,
            log_figure=True,
            task=task,
            num_classes=num_classes,
        ),
    ]

    metrics = [metric for metric in default_metrics if metric.name in metrics_list]
    return metrics


class EarlyStopping:
    """Early stopping mechanism.

    Takes effect if the validation loss does not improve after a given number of validation periods.

    Adapted from https://github.com/Bjarten/early-stopping-pytorch.

    Args:
        patience: How long to wait after last time validation loss improved.
            This corresponds to the number of validation periods which
            can either be batches or epochs.
            Default: 5
        delta: Minimum change in the monitored quantity to qualify as an improvement.
            Default: 0
    """

    def __init__(
        self,
        patience: int | None,
        delta: int = 0,
    ):
        self.patience = patience if patience else 5
        self.counter = 0
        self.best_score = -np.inf
        self.delta = delta

    def __call__(self, val_loss: float, model: Model) -> bool:
        """Call Early Stopping.

        Args:
            val_loss: Current validation loss to evaluate.
            model: Current model to save.

        Returns:
            Whether to stop early.
        """
        score = -val_loss

        if self.best_score == -np.inf:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(
                    f"EarlyStopping: Validation loss did not decrease "
                    f"for {self.counter} validation periods."
                )

                return True
        else:
            self.best_score = score
            self.counter = 0
        return False
