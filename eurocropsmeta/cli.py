import logging

import eurocropsml.settings
import torch.multiprocessing
import typer

from eurocropsmeta.experiment.cli import experiments_app
from eurocropsmeta.settings import ROOT_DIR

# if this CLI is run we use it for determining the overall root directory
eurocropsml.settings.ROOT_DIR = ROOT_DIR

# Fix `too many open files` error
torch.multiprocessing.set_sharing_strategy("file_system")

logger = logging.getLogger(__name__)

cli = typer.Typer(name="EuroCrops Meta-Learning")


@cli.callback()
def logging_setup() -> None:
    """Logging setup for CLI."""

    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )


cli.add_typer(experiments_app)


if __name__ == "__main__":
    cli()
