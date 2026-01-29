import json
import logging
import os
from typing import Literal, Optional, Type, TypeVar

import typer
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import BaseModel

from eurocropsmeta.experiment.eurocrops.config import EuroCropsTransferConfig
from eurocropsmeta.experiment.eurocrops.experiment import EuroCropsExperimentBuilder
from eurocropsmeta.experiment.transfer import (
    TransferExperiment,
    TransferExperimentBuilder,
)
from eurocropsmeta.settings import Settings

logger = logging.getLogger(__name__)

experiments_app = typer.Typer(name="experiments")

ConfigT = TypeVar("ConfigT", bound=BaseModel)
BuilderT = TypeVar("BuilderT", bound=TransferExperimentBuilder)
OverridesT = Optional[list[str]]  # typer requires the old Optional[...] annotation


def build_experiment_app(
    experiment_name: str, config_class: Type[ConfigT], builder_class: Type[BuilderT]
) -> typer.Typer:
    """Build a CLI component for running experiments."""

    config_dir = Settings().experiment_dir.joinpath(experiment_name, "config")

    app = typer.Typer(name=experiment_name)

    def build_config(overrides: OverridesT) -> ConfigT:
        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
            if overrides is None:
                overrides = []
            # Change working directory to resolve additional config paths
            os.chdir(Settings().experiment_dir)
            composed_config = compose(config_name="config", overrides=overrides)
        config = config_class(**OmegaConf.to_object(composed_config))  # type: ignore[arg-type]
        return config

    def build_experiment(
        config: ConfigT, mode: Literal["pretrain", "finetune"]
    ) -> TransferExperiment:
        builder: TransferExperimentBuilder = builder_class(config)
        return builder.build_experiment(mode=mode)

    @app.command(name="config")
    def print_config(
        overrides: OverridesT = typer.Argument(None, help="Overrides to the experiment config."),
    ) -> None:
        """Print the currently used config."""
        config = build_config(overrides)
        print(OmegaConf.to_yaml(json.loads(config.model_dump_json())))

    @app.command()
    def pretrain_tuning(
        study_name: str = typer.Option(
            None,
            help="Name of the tuning study." "All runs will contain this identifier in their name.",
        ),
        overrides: OverridesT = typer.Argument(None, help="Overrides to the experiment config."),
    ) -> None:
        """Run the hyperparameter tuning for pretraining."""
        experiment = build_experiment(build_config(overrides), mode="pretrain")
        experiment.run_pretrain_tuning(study_name=study_name)

    @app.command()
    def pretraining(
        run_name: str = typer.Option(
            None, help="Name of the experiment run. Will use tuning with same name."
        ),
        overrides: OverridesT = typer.Argument(None, help="Overrides to experiment config."),
    ) -> None:
        """Run pretraining."""
        experiment = build_experiment(build_config(overrides), mode="pretrain")
        experiment.run_pretraining(run_name=run_name)

    @app.command()
    def tuned_pretraining(
        run_name: str = typer.Option(None, help="Name of the experiment run."),
        overrides: OverridesT = typer.Argument(None, help="Overrides to the experiment config."),
    ) -> None:
        """Run hyperparameter tuning followed by pretraining."""
        pretrain_tuning(run_name, overrides)
        pretraining(run_name, overrides)

    @app.command()
    def finetune_tuning(
        task: str = typer.Argument(..., help="Task to finetune on."),
        study_name: str = typer.Option(
            None,
            help="Name of the tuning study." "All runs will contain this identifier in their name.",
        ),
        pretrain_run: str = typer.Option(
            "auto",
            help="Name of the pretraining run to start from. "
            "Either 'auto', 'random' or a run name.",
        ),
        overrides: OverridesT = typer.Argument(None, help="Overrides to the experiment config."),
    ) -> None:
        """Run the hyperparameter tuning for finetuning."""
        if pretrain_run == "random":
            pretrain_run_name = None
        elif pretrain_run == "auto" and study_name is not None:
            pretrain_run_name = study_name
        else:
            pretrain_run_name = pretrain_run
        experiment = build_experiment(build_config(overrides), mode="finetune")
        experiment.run_finetune_tuning(
            task_name=task, study_name=study_name, pretrain_run_name=pretrain_run_name
        )

    @app.command()
    def finetuning(
        task: str = typer.Argument(..., help="Task to finetune on."),
        run_name: str = typer.Option(None, help="Name of the experiment run."),
        pretrain_run: str = typer.Option(
            "auto",
            help="Name of the pretraining run to start from. "
            "Either 'auto', 'random' or a run name.",
        ),
        overrides: OverridesT = typer.Argument(None, help="Overrides to the experiment config."),
    ) -> None:
        """Run the finetuning."""
        if pretrain_run == "random":
            pretrain_run_name = None
        elif pretrain_run == "auto" and run_name is not None:
            pretrain_run_name = run_name
        else:
            pretrain_run_name = pretrain_run
        experiment = build_experiment(build_config(overrides), mode="finetune")
        experiment.run_finetuning(
            task_name=task, run_name=run_name, pretrain_run_name=pretrain_run_name
        )

    @app.command()
    def tuned_finetuning(
        task: str = typer.Argument(..., help="Task to finetune on."),
        run_name: str = typer.Option(None, help="Name of the experiment run."),
        pretrain_run: str = typer.Option(
            "auto",
            help="Name of the pretraining run to start from. "
            "Either 'auto', 'random' or a run name.",
        ),
        overrides: OverridesT = typer.Argument(None, help="Overrides to the experiment config."),
    ) -> None:
        """Run hyperparameter tuning followed by finetuning."""
        finetune_tuning(task, run_name, pretrain_run, overrides)
        finetuning(task, run_name, pretrain_run, overrides)

    return app


eurocrops_app = build_experiment_app(
    "eurocrops", EuroCropsTransferConfig, EuroCropsExperimentBuilder
)

experiments_app.add_typer(eurocrops_app)
