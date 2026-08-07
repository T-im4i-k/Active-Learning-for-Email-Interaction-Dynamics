from abc import ABC, abstractmethod

class AbstractTemplateWeight(ABC):
    def __init__(self, min_template_id: int, max_template_id: int) -> None:
        self.min_template_id: int = min_template_id
        self.max_template_id: int = max_template_id

    @abstractmethod
    def _raw_weight(self, template_id: int) -> float:
        """Unnormalized recency weight, anchored to the global template timeline."""
        raise NotImplementedError

    def __call__(self, template_id: int) -> float:
        if not self.min_template_id <= template_id <= self.max_template_id:
            raise ValueError("Template ID is out of bounds")
        return self._raw_weight(template_id)


class ExponentialTemplateWeight(AbstractTemplateWeight):
    def __init__(self, min_template_id: int, max_template_id: int, half_life: float) -> None:
        super().__init__(min_template_id, max_template_id)

        # if not 0 < half_life <= 1:
        #     raise ValueError("Half life is out of bounds")

        self.half_life: float = half_life

    def _raw_weight(self, template_id: int) -> float:
        delta_id: float = (self.max_template_id - template_id)/(self.max_template_id - self.min_template_id)
        return 2 ** (-delta_id / self.half_life)

class UniformTemplateWeight(AbstractTemplateWeight):
    def _raw_weight(self, template_id: int) -> float:
        return 1.0


class TemplateWeightFactory:
    @classmethod
    def from_config(cls, config: dict) -> AbstractTemplateWeight:
        template_weight: str = config["template_weight"]

        match template_weight:
            case "uniform":
                return UniformTemplateWeight(min_template_id=config["min_template_id"],max_template_id=config["max_template_id"])
            case "exponential":
                return ExponentialTemplateWeight(
                    min_template_id=config["min_template_id"],
                    max_template_id=config["max_template_id"],
                    half_life=config["template_weight_params"]["half_life"]
                )