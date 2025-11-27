from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).parents[1]


class Settings(BaseSettings):
    """Global settings.

    Args:
        experiment_dir: Directory in which experiment runs are stored.
        mlflow_uri: URI to mlflow tracking server.
        disable_cudnn: Whether to disable cuDNN.
            This may be necessary when training RNNs.
    """

    experiment_dir: Path = Field(
        Path("experiments"), validation_alias="EUROCROPS_META_LEARNING_EXPERIMENT_DIR"
    )
    mlflow_uri: str = Field(
        "file:./mlruns",  # Local file storage by default
        validation_alias="EUROCROPS_META_LEARNING_MLFLOW_URI",
    )

    disable_cudnn: bool = Field(False, validation_alias="EUROCROPS_META_LEARNING_DISABLE_CUDNN")
    seed: int = Field(42, validation_alias="EUROCROPS_META_LEARNING_SEED")

    @field_validator("experiment_dir")
    @classmethod
    @classmethod
    def relative_path(cls, v: Path) -> Path:
        """Interpret relative paths w.r.t. the project root."""
        if not v.is_absolute():
            v = ROOT_DIR.joinpath(v)
        return v

    @field_validator("mlflow_uri")
    @classmethod
    @classmethod
    def parse_mlflow_uri(cls, v: str) -> str:
        """Interpret relative paths w.r.t. the project root."""
        if not v.startswith("http"):
            v = "file://" + str(Path(v).absolute())
        return v
