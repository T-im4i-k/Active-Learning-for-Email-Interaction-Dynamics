from abc import ABC, abstractmethod

import numpy as np


class AbstractTTODecay(ABC):
    @abstractmethod
    def __call__(self, tto: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ExponentialTTODecay(AbstractTTODecay):
    def __init__(self, half_life: float):
        if half_life <= 0:
            raise ValueError(f"half_life must be strictly positive, got {half_life}")
        self.half_life = half_life

    def __call__(self, tto: np.ndarray) -> np.ndarray:
        return np.power(2.0, -tto / self.half_life)


class HyperbolicTTODecay(AbstractTTODecay):
    def __init__(self, dumping: float):
        if dumping <= 0:
            raise ValueError(f"dumping must be strictly positive, got {dumping}")
        self.dumping = dumping

    def __call__(self, tto: np.ndarray) -> np.ndarray:
        return self.dumping / (tto + self.dumping)



class TTODecayFactory:
    @classmethod
    def from_config(cls, config: dict) -> AbstractTTODecay | None:
        decay_config = config["tto_decay"]
        if len(decay_config) == 0:
            return None

        decay_type = decay_config["type"]

        match decay_type:
            case "exponential":
                return ExponentialTTODecay(half_life=decay_config["half_life"])
            case "hyperbolic":
                return HyperbolicTTODecay(dumping=decay_config["dumping"])
            case _:
                raise ValueError(f"Unknown decay type: {decay_type}")
