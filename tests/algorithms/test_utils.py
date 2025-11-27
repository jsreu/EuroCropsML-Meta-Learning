import torch
from torch import nn

from eurocropsmeta.algorithms.utils import clone_module, update_module


class SequentialModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.sequential = nn.Sequential(self.linear, self.linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)  # type: ignore[no-any-return]


def test_clone_module_forward() -> None:
    module = SequentialModule()
    cloned = clone_module(module)

    data = torch.randn((10, 1))
    module_out = module(data)
    cloned_out = cloned(data)

    assert (module_out == cloned_out).all()


def test_clone_module_backward() -> None:
    module = SequentialModule()
    cloned = clone_module(module)

    data = torch.randn((10, 1))
    cloned_out = cloned(data)

    expected_grad = 2 * module.linear.weight * data.sum()

    loss = cloned_out.sum()
    loss.backward()

    assert torch.allclose(module.linear.weight.grad, expected_grad)  # type: ignore[arg-type]


# Check that duplicate clones are avoided
def test_clone_module_cloned_size() -> None:
    module = SequentialModule()
    cloned = clone_module(module)

    module_sum = sum(sum(p.shape) for p in module.parameters())
    cloned_sum = sum(sum(p.shape) for p in cloned.parameters())
    assert module_sum == cloned_sum


def test_update_module_forward() -> None:
    module = SequentialModule()
    cloned = clone_module(module)

    alpha = torch.tensor(2.0)
    for p in cloned.parameters():
        p.update = alpha  # type: ignore[attr-defined]
    updated = update_module(cloned)
    for p1, p2 in zip(updated.parameters(), cloned.parameters()):
        assert (p1 == p2).all()
    assert cloned(torch.ones(1)) == (module.linear.weight + alpha) ** 2


def test_update_module_backward() -> None:
    module = SequentialModule()
    cloned = clone_module(module)

    data = torch.randn((10, 1))
    cloned_out = cloned(data)

    alpha = torch.tensor(2.0)
    for p in cloned.parameters():
        p.update = alpha  # type: ignore[attr-defined]
    _ = update_module(cloned)

    data = torch.randn((10, 1))
    cloned_out = cloned(data)
    loss = cloned_out.sum()
    loss.backward()

    expected_grad = 2 * (module.linear.weight + alpha) * data.sum()
    assert torch.allclose(module.linear.weight.grad, expected_grad)  # type: ignore[arg-type]
