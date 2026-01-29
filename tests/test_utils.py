import pytest

from eurocropsmeta.utils import BaseConfig


class ExampleConfig(BaseConfig):
    example_int_field: int
    example_float_field: float
    example_str_field: str


@pytest.fixture
def config() -> ExampleConfig:
    return ExampleConfig(
        example_int_field=1,
        example_float_field=2.0,
        example_str_field="3.0",
    )


def test_config(config: ExampleConfig) -> None:
    schema = config.params()
    assert set(schema) == {
        "example_int_field",
        "example_float_field",
        "example_str_field",
    }
