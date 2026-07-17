from pathlib import Path
from dataclasses import dataclass

DEFAULT_CONFIG_NAME = "config.json"


@dataclass
class ExperimentFolder:
    path: Path
    config_name: str = DEFAULT_CONFIG_NAME

    @property
    def config_path(self) -> Path:
        return self.path / self.config_name

    @classmethod
    def from_config_path(cls, config_path: Path) -> "ExperimentFolder":
        return cls(path=config_path.parent, config_name=config_path.name)
