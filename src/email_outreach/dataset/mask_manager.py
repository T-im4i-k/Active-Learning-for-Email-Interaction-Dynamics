import torch
from dataclasses import dataclass


def _sample_bernoulli(shape: tuple[int, ...], p: float, gen: torch.Generator | None = None) -> torch.Tensor:
    return torch.bernoulli(
        torch.ones(shape) * p,
        generator=gen
    ).bool()


@dataclass(slots=True, frozen=True)
class MaskManager:
    train_mask: torch.Tensor
    val_mask: torch.Tensor

    @classmethod
    def sample(cls, shape: tuple[int, ...], train_p: float, val_p: float,
               gen: torch.Generator | None = None) -> "MaskManager":
        val_mask: torch.Tensor = _sample_bernoulli(shape, p=val_p, gen=gen)
        train_mask: torch.Tensor = _sample_bernoulli(shape, p=train_p, gen=gen) & val_mask
        return cls(train_mask, val_mask)