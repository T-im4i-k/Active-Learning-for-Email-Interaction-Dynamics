import json
from email_outreach.utils.experiment_folder import ExperimentFolder
from dataclasses import dataclass

from typing import Any


@dataclass
class JsonConfig:
    data: dict[str, Any]

    @classmethod
    def load(cls, experiment_folder: ExperimentFolder) -> "JsonConfig":
        with open(experiment_folder.config_path, "r") as file:
            data: dict[str, Any] = json.load(file)
            return cls(data)

    def save(self, experiment_folder: ExperimentFolder) -> None:
        experiment_folder.path.mkdir(parents=True, exist_ok=True)
        with open(experiment_folder.config_path, "w") as file:
            json.dump(self.data, file, indent=4)
