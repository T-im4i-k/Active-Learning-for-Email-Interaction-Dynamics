from abc import ABC, abstractmethod
import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class CoefSchedulerArgs:
    batch: int
    open_frac: float


class AbstractCoefScheduler(ABC):
    @abstractmethod
    def __call__(self, args: CoefSchedulerArgs) -> float:
        raise NotImplementedError


class ConstantCoefScheduler(AbstractCoefScheduler):
    def __init__(self, coef: float) -> None:
        if not 0.0 <= coef <= 1.0:
            raise ValueError("Argument coef must be in range [0, 1]")
        self.coef: float = coef

    def __call__(self, args: CoefSchedulerArgs) -> float:
        if args.batch < 0:
            raise ValueError("Argument batch must be a non-negative integer")
        return self.coef


class AdaptiveCoefScheduler(AbstractCoefScheduler):
    def __init__(self, dumping: float):
        self.dumping: float = dumping

    def __call__(self, args: CoefSchedulerArgs) -> float:
        return self.dumping / (args.open_frac + self.dumping)


class InterpolatingCoefScheduler(AbstractCoefScheduler, ABC):
    def __init__(
            self, left_bound: float, right_bound: float, batch_sizes: Sequence[float]
    ) -> None:
        if not 0.0 <= left_bound <= 1.0:
            raise ValueError("Argument left_bound must be in range [0, 1]")

        if not 0.0 <= right_bound <= 1.0:
            raise ValueError("Argument right_bound must be in range [0, 1]")

        if len(batch_sizes) < 2:
            raise ValueError(
                "Argument batch_sizes must contain at least 2 entries "
                "(at least one active-learning batch plus the final batch)"
            )

        if any(size <= 0 for size in batch_sizes):
            raise ValueError("All entries of batch_sizes must be positive integers")

        self.left_bound: float = left_bound
        self.right_bound: float = right_bound
        self.batch_sizes: Sequence[float] = batch_sizes
        self.batches: int = len(batch_sizes)
        self._cumulative_sizes: list[float] = list(itertools.accumulate(batch_sizes))
        self._total_emails: float = self._cumulative_sizes[-1]

    def _ratio(self, batch: int) -> float:
        if not 0 <= batch < len(self.batch_sizes):
            raise ValueError("Argument batch must be an integer in range [0, batches)")

        return self._cumulative_sizes[batch] / self._total_emails


class GeometricCoefScheduler(InterpolatingCoefScheduler):
    def __call__(self, args: CoefSchedulerArgs) -> float:
        ratio: float = self._ratio(args.batch)
        return self.left_bound ** (1 - ratio) * self.right_bound ** ratio


class LinearCoefScheduler(InterpolatingCoefScheduler):
    def __call__(self, args: CoefSchedulerArgs) -> float:
        ratio: float = self._ratio(args.batch)
        return self.left_bound * (1 - ratio) + self.right_bound * ratio


class AbstractSchedulerFactory(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> AbstractCoefScheduler | None:
        raise NotImplementedError


class AlphaSchedulerFactory(AbstractSchedulerFactory):
    @classmethod
    def from_config(cls, config: dict) -> AbstractCoefScheduler:
        scheduler_type: str = config["alpha_type"].lower()
        alpha_params: dict = config["alpha_params"]

        normal_batch_size = config["sent_by_T"] / config["num_splits"]
        batch_sizes: Sequence[float] = [normal_batch_size] * config["num_splits"] + [config["sent_after_T"]]

        match scheduler_type:
            case "constant":
                return ConstantCoefScheduler(coef=alpha_params["alpha"])
            case "geometric":
                return GeometricCoefScheduler(
                    left_bound=alpha_params["left_bound"],
                    right_bound=alpha_params["right_bound"],
                    batch_sizes=batch_sizes
                )
            case "linear":
                return LinearCoefScheduler(
                    left_bound=alpha_params["left_bound"],
                    right_bound=alpha_params["right_bound"],
                    batch_sizes=batch_sizes
                )
            case "adaptive":
                return AdaptiveCoefScheduler(
                    dumping=alpha_params["dumping"]
                )
            case _:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class BetaSchedulerFactory(AbstractSchedulerFactory):
    @classmethod
    def from_config(cls, config: dict) -> AbstractCoefScheduler | None:
        if config["beta_type"] is None:
            return None

        scheduler_type: str = config["beta_type"].lower()
        beta_params: dict = config["beta_params"]

        normal_batch_size = config["sent_by_T"] / config["num_splits"]
        batch_sizes: Sequence[float] = [normal_batch_size] * config["num_splits"] + [config["sent_after_T"]]

        match scheduler_type:
            case "constant":
                return ConstantCoefScheduler(coef=beta_params["beta"])
            case "geometric":
                return GeometricCoefScheduler(
                    left_bound=beta_params["left_bound"],
                    right_bound=beta_params["right_bound"],
                    batch_sizes=batch_sizes
                )
            case "linear":
                return LinearCoefScheduler(
                    left_bound=beta_params["left_bound"],
                    right_bound=beta_params["right_bound"],
                    batch_sizes=batch_sizes
                )
            case "adaptive":
                return AdaptiveCoefScheduler(
                    dumping=beta_params["dumping"]
                )
            case _:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")
