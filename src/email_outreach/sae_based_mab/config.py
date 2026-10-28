import csv
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

import pandas as pd


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("true", "1", "1.0"):
        return True
    if v in ("false", "0", "0.0", ""):
        return False
    raise ValueError(f"Cannot parse bool from {value!r}")


@dataclass
class ContextualBanditMetrics:
    mailshot_id: int
    opened: bool
    prediction: float

    present_in_prediction: bool
    stage: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "mailshot_id": (int(self.mailshot_id)),
            "present_in_prediction": bool(self.present_in_prediction),
            "opened": bool(self.opened),
            "prediction": (float(self.prediction)),
            "stage": int(self.stage),
        }

    @staticmethod
    def to_csv_gz(
        metrics: list, folder: Path, filename: str = "results.csv.gz"
    ) -> None:
        metrics_dicts = [m.to_dict() for m in metrics]
        df = pd.DataFrame(metrics_dicts)
        df.to_csv(folder / filename, compression="gzip", index=False)

    @classmethod
    def from_csv_gz(cls, folder: Path, filename: str = "results.csv.gz") -> list[Self]:
        path = folder / filename
        if not path.exists():
            raise FileNotFoundError(path)

        instances: list[ContextualBanditMetrics] = []

        with gzip.open(path, mode="rt", newline="") as fh:  # text mode
            reader = csv.DictReader(fh)
            for row in reader:
                instances.append(
                    cls(
                        mailshot_id=int(row["mailshot_id"]),
                        present_in_prediction=_parse_bool(row["present_in_prediction"]),
                        opened=_parse_bool(row["opened"]),
                        prediction=float(row["prediction"]),
                        stage=(
                            int(row["stage"])
                            if row.get("stage") not in ("", None, "None")
                            else -1
                        ),
                    )
                )

        return instances

    @classmethod
    def from_lists(
        cls,
        mailshot_id: int,
        present_in_prediction: list[bool],
        opened: list[bool],
        prediction: list[float],
    ) -> list[Self]:
        return [
            cls(mailshot_id, o, pred, p)
            for p, o, pred in zip(present_in_prediction, opened, prediction)
        ]


@dataclass
class ContextualBanditWithAutoencoderConfig:
    d: int = 16
    epochs: int = 30
    lr: float = 0.01
    wd: float = 0.0001
    batch_size: int = 16
    positive_weight: float = 100.0
    alpha_params: dict[str, Any] = field(
        default_factory=lambda: {"type": "constant", "alpha": 0.1}
    )
    beta_params: dict[str, Any] = field(default_factory=lambda: {})
    template_weight_params: dict[str, Any] = field(
        default_factory=lambda: {"type": "uniform"}
    )
    G: float = 10000.0
    naive_f: bool = True
    revised_s: bool = False
    T: int = 2880
    num_splits: int = 24
    sent_by_T: float = 0.15
    sent_after_T: float = 0.1
    j: int = 0
    dropout: float = 0.0
    p_type: Literal["mean", "var", "std"] = "mean"
    noise: float = 0.0
    double_sae: bool = False
    negative_weight: float = 0.1
    f_confidence: float = 1.0
    negative_feedback_ratio: float = 0.1
    legacy_sae: bool = True

    def to_dict(self):
        return {
            "d": self.d,
            "epochs": self.epochs,
            "lr": self.lr,
            "wd": self.wd,
            "batch_size": self.batch_size,
            "positive_weight": self.positive_weight,
            "alpha_params": self.alpha_params,
            "beta_params": self.beta_params,
            "template_weight_params": self.template_weight_params,
            "G": self.G,
            "naive_f": self.naive_f,
            "revised_s": self.revised_s,
            "num_splits": self.num_splits,
            "T": self.T,
            "sent_by_T": self.sent_by_T,
            "sent_after_T": self.sent_after_T,
            "dropout": self.dropout,
            "j": self.j,
            "p_type": self.p_type,
            "noise": self.noise,
            "double_sae": self.double_sae,
            "negative_weight": self.negative_weight,
            "f_confidence": self.f_confidence,
            "negative_feedback_ratio": self.negative_feedback_ratio,
            "legacy_sae": self.legacy_sae,
        }

    @classmethod
    def from_json_file(cls, folder: Path, filename: str = "config.json") -> Self:
        with open(folder / filename, "r") as f:
            data = json.load(f)

        return cls(**data)

    def to_json_file(self, folder: Path, filename: str = "config.json") -> None:
        with open(folder / filename, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
