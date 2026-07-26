from abc import ABC, abstractmethod
import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AlphaSchedulerArgs:
    batch: int
    open_frac: float


class AbstractAlphaScheduler(ABC):
    @abstractmethod
    def __call__(self, args: AlphaSchedulerArgs) -> float:
        raise NotImplementedError


class ConstantAlphaScheduler(AbstractAlphaScheduler):
    def __init__(self, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Argument alpha must be in range [0, 1]")
        self.alpha: float = alpha

    def __call__(self, args: AlphaSchedulerArgs) -> float:
        if args.batch < 0:
            raise ValueError("Argument batch must be a non-negative integer")
        return self.alpha


class AdaptiveAlphaScheduler(AbstractAlphaScheduler):
    def __init__(self, dumping: float):
        self.dumping: float = dumping

    def __call__(self, args: AlphaSchedulerArgs) -> float:
        return self.dumping / (args.open_frac + self.dumping)


class InterpolatingAlphaScheduler(AbstractAlphaScheduler, ABC):
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


class GeometricAlphaScheduler(InterpolatingAlphaScheduler):
    def __call__(self, args: AlphaSchedulerArgs) -> float:
        ratio: float = self._ratio(args.batch)
        return self.left_bound ** (1 - ratio) * self.right_bound ** ratio


class LinearAlphaScheduler(InterpolatingAlphaScheduler):
    def __call__(self, args: AlphaSchedulerArgs) -> float:
        ratio: float = self._ratio(args.batch)
        return self.left_bound * (1 - ratio) + self.right_bound * ratio


# class SigmoidAlphaScheduler(InterpolatingAlphaScheduler):
#     def __init__(self, left_bound: float, right_bound: float, batches: int, slope_coef: float,
#                  shift_coef: float) -> None:
#         if slope_coef < 0:
#             raise ValueError("Argument slope_coef must be in non-negative")
#
#         super().__init__(left_bound, right_bound, batches)
#         self.slope_coef: float = slope_coef
#         self.shift_coef: float = shift_coef
#
#     def __call__(self, batch: int) -> float:
#         if not 0 <= batch <= self.batches:
#             raise ValueError("Argument batch must be an integer in range [0, batches]")
#
#         ratio: float = batch / self.batches
#         coef: float = expit(self.slope_coef * (ratio - self.shift_coef))
#         return self.left_bound * (1 - coef) + self.right_bound * coef


class AlphaSchedulerFactory:
    @classmethod
    def from_config(cls, config: dict) -> AbstractAlphaScheduler:
        scheduler_type: str = config["alpha_type"].lower()
        alpha_params: dict = config["alpha_params"]

        normal_batch_size = config["sent_by_T"] / config["num_splits"]
        batch_sizes: Sequence[float] = [normal_batch_size] * config["num_splits"] + [config["sent_after_T"]]

        match scheduler_type:
            case "constant":
                return ConstantAlphaScheduler(alpha=alpha_params["alpha"])
            case "geometric":
                return GeometricAlphaScheduler(
                    left_bound=alpha_params["left_bound"],
                    right_bound=alpha_params["right_bound"],
                    batch_sizes=batch_sizes
                )
            case "linear":
                return LinearAlphaScheduler(
                    left_bound=alpha_params["left_bound"],
                    right_bound=alpha_params["right_bound"],
                    batch_sizes=batch_sizes
                )
            case "adaptive":
                return AdaptiveAlphaScheduler(
                    dumping=alpha_params["dumping"]
                )
            # case "sigmoid":
            #     return SigmoidAlphaScheduler(
            #         left_bound=alpha_params["left_bound"],
            #         right_bound=alpha_params["right_bound"],
            #         batches=config["num_splits"],
            #         slope_coef=alpha_params["slope_coef"],
            #         shift_coef=alpha_params["shift_coef"],
            #     )

            case _:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")
