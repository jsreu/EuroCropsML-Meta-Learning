from eurocropsmeta.experiment.utils import recursive_update


def test_recursive_update() -> None:
    d = {"a": {"b": {"c": 1}}, "d": 2}
    recursive_update(d, "c", 2)
    assert d == {"a": {"b": {"c": 2}}, "d": 2}

    recursive_update(d, "d", 10)
    assert d == {"a": {"b": {"c": 2}}, "d": 10}
