import logging
from collections.abc import Sequence
from itertools import islice

import torch
from tqdm import tqdm

from eurocropsmeta.algorithms.base import MetaLearnAlgorithm
from eurocropsmeta.dataset.task import Task, TaskDataset
from eurocropsmeta.models.base import Model
from eurocropsmeta.train.utils import ScalarMetric, TaskMetric

from .callback import TrainCallback

logger = logging.getLogger(__name__)


class MetaTrainer:
    """Helper class for running meta learning algorithms.

    Args:
        meta_learner: Meta learning algorithm to use for training.
        callbacks: List of callbacks used during training.
        prefetch_task_data: If not None, prefetch the specified amount
            of task data for faster data loading.
    """

    def __init__(
        self,
        meta_learner: MetaLearnAlgorithm,
        callbacks: Sequence[TrainCallback] | None = None,
        prefetch_task_data: int | None = None,
    ):
        self.meta_learner = meta_learner
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks
        self.prefetch_task_data = prefetch_task_data

    def train(
        self,
        train_set: TaskDataset,
        val_set: TaskDataset | None = None,
        validate_every: int = 10,
    ) -> Model:
        """Adapt model via given meta learning algorithm.

        Args:
            train_set: TaskDataset to use for meta-training.
            val_set: (Optional) task validation set to use.
            validate_every: Interval between validation runs.

        Returns:
            Adapted model

        Raises:
            ValueError: If the validation set has infinite size.
        """
        if val_set:
            logger.info("Preparing validation tasks.")
            num_val_tasks = val_set.num_tasks()
            if num_val_tasks is None:
                raise ValueError("Given validation set has infinite size.")
            val_tasks = _generate_tasks(val_set)
        else:
            val_tasks = []

        if self.prefetch_task_data:
            num_workers = None
            # Fix `too many open files` error
            torch.multiprocessing.set_sharing_strategy("file_system")
        else:
            num_workers = 0
        task_dl = train_set.dataloader(
            batch_size=self.meta_learner.config.tasks_per_batch,
            prefetch_task_data=self.prefetch_task_data or False,
            num_workers=num_workers,
        )
        total_batches = self.meta_learner.config.total_task_batches
        train_iter = tqdm(islice(task_dl, total_batches), total=total_batches, desc="Meta-training")

        for step, tasks in enumerate(train_iter):
            if not tasks:
                logger.warning("Empty task batch.")
                continue
            outer_loss = self.meta_learner.adapt(tasks)
            train_metrics: Sequence[TaskMetric] = []
            for callback in self.callbacks:
                callback.train_callback(
                    outer_loss, train_metrics, self.meta_learner.model, step=step
                )
            if val_tasks and step and step % validate_every == 0:
                val_loss, val_metrics = self._eval_model(val_tasks)
                for callback in self.callbacks:
                    callback.validation_callback(
                        val_loss,
                        val_metrics,
                        self.meta_learner.model,
                        step=step,
                    )
        for callback in self.callbacks:
            callback.end_callback(self.meta_learner.model)
        return self.meta_learner.model

    def test(self, test_set: TaskDataset) -> tuple[float, Sequence[TaskMetric]]:
        """Test adapted model on test set with given meta-learning algorithm."""

        logger.info("Preparing test tasks.")
        test_tasks = _generate_tasks(test_set)
        test_loss, test_metrics = self._eval_model(test_tasks)
        for callback in self.callbacks:
            callback.test_callback(
                test_loss,
                test_metrics,
                self.meta_learner.model,
                step=None,
            )

        return test_loss, test_metrics

    def _eval_model(self, test_tasks: list[Task]) -> tuple[float, Sequence[ScalarMetric]]:
        task_losses: list[float] = []
        task_metrics_dicts: list[dict[str, ScalarMetric]] = []
        for task in test_tasks:
            task_loss, task_metrics, _ = self.meta_learner.eval(task)
            task_losses.append(task_loss)
            task_metrics_dicts.append(
                {metric.name: metric for metric in task_metrics if isinstance(metric, ScalarMetric)}
            )

        avg_loss = sum(loss for loss in task_losses) / len(test_tasks)
        avg_metrics_dict: dict[str, ScalarMetric] = {}
        for metrics_dict in task_metrics_dicts:
            for name, metric in metrics_dict.items():
                if name in avg_metrics_dict:
                    avg_metrics_dict[name] += metric
                else:
                    avg_metrics_dict[name] = metric

        avg_metrics: Sequence[ScalarMetric] = [
            metric / len(test_tasks) for metric in avg_metrics_dict.values()
        ]
        return avg_loss, avg_metrics


def _generate_tasks(task_set: TaskDataset, pbar: bool = True) -> list[Task]:
    num_val_tasks = task_set.num_tasks()
    if num_val_tasks is None:
        raise ValueError("Given validation set has infinite size.")
    if num_val_tasks > 1e9:
        raise ValueError("Number of test tasks is too large.")
    tasks = list(
        tqdm(
            (task for tasks in task_set.dataloader(batch_size=1, num_workers=0) for task in tasks),
            total=num_val_tasks,
            disable=not pbar,
        )
    )
    return tasks
