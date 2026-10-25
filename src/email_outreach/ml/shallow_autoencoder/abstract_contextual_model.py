import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Sequence,
    Type,
)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterSampler

from email_outreach.experiment_utils.experiment_constants import (
    EXPERIMENT_RESULTS_FOLDER_NAME,
    get_experiment_folder_path,
    get_test_folder_path,
)
from email_outreach.ml.shallow_autoencoder.dataset.autoencoder_dataset import (
    AutoencoderDataset,
)

from email_outreach.ml.shallow_autoencoder.config import (
    ContextualBanditMetrics,
    ContextualBanditWithAutoencoderConfig,
)

logger = logging.getLogger(__name__)


class AbstractContextualModel(ABC):

    def __init__(self, config):
        self.config = config
        self._train = None
        self._val = None

    @property
    def train(self) -> AutoencoderDataset:
        if self._train is None:
            raise ValueError("Train dataset not set")
        return self._train

    @property
    def val(self) -> AutoencoderDataset:
        if self._val is None:
            raise ValueError("Validation dataset not set")
        return self._val

    @abstractmethod
    def fit(self, train: AutoencoderDataset, val: AutoencoderDataset) -> None:
        self._train = train
        self._val = val

    @abstractmethod
    def predict(self, data: AutoencoderDataset) -> list[ContextualBanditMetrics]:
        pass

    @classmethod
    @abstractmethod
    def model_name(cls) -> str:
        raise NotImplementedError

    @classmethod
    def config_class(cls) -> Type[ContextualBanditWithAutoencoderConfig]:
        return ContextualBanditWithAutoencoderConfig

    @classmethod
    @abstractmethod
    def metrics_class(cls) -> Type[ContextualBanditMetrics]:
        raise NotImplementedError

    @classmethod
    def test(
        cls,
        sender_id: int,
        split_sizes: list[int],
        repetitions: int,
        version: str,
        n_jobs,
    ) -> None:
        if len(split_sizes) != 1:
            raise ValueError("Only one split size is allowed for this experiment.")

        split_size = split_sizes[0]
        experiments_folder = get_test_folder_path(
            sender_id=sender_id, model_name=cls.model_name(), version=version
        )
        config: ContextualBanditWithAutoencoderConfig = (
            cls.config_class().from_json_file(experiments_folder)
        )

        Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(cls._run_single_test)(
                sender_id=sender_id,
                repetitions=repetitions,
                split_size=split_size,
                experiments_folder=experiments_folder,
                config=config,
                repetition=repetition,
                i=i,
                current_split_size=current_split_size,
            )
            for repetition in range(repetitions)
            for i, current_split_size in enumerate(range(split_size, 0, -1))
        )

    @classmethod
    def _run_single_test(
        cls,
        sender_id,
        repetitions,
        split_size,
        experiments_folder,
        config,
        repetition,
        i,
        current_split_size,
    ):
        logger.info(
            f"Experiment {i + 1}/{split_size} - Repetition {repetition + 1}/{repetitions}"
        )
        # Load datasets using the provided sender_id and split configuration
        # Always have size 1 for the test set, the last split is not used
        datasets = AutoencoderDataset.from_disk_data_split(
            sender_id,
            split_sizes=[1, current_split_size - 1],
            remove_users_below_n_opens=1,
        )
        train, test = datasets[0], datasets[1]
        model = cls(config=config)
        model.fit(train, test)
        metrics: list[ContextualBanditMetrics] = model.predict(test)

        # Create a folder for the current split size and repetition
        current_folder = experiments_folder / (
            EXPERIMENT_RESULTS_FOLDER_NAME + f"_{i + 1}_{repetition}"
        )
        current_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Current folder: {current_folder}. Saving results...")
        # Save the prediction metrics
        ContextualBanditMetrics.to_csv_gz(metrics, folder=current_folder)
        config.to_json_file(folder=current_folder)

    @classmethod
    def grid_search(
        cls,
        sender_id: int,
        split_sizes: list[int],
        version: str,
        param_grid: dict[str, list[Any]],
        n_samples: int | None = None,
        n_jobs: int = 8,
        random_state: int = 42,
    ) -> None:

        logger.info(
            f"Starting grid search for {cls.model_name()} with {len(param_grid)} parameters"
        )

        # Create the results folder
        results_folder = get_experiment_folder_path(
            sender_id, cls.model_name(), version
        )
        test_folder = get_test_folder_path(
            sender_id=sender_id, model_name=cls.model_name(), version=version
        )

        results_folder.mkdir(parents=True, exist_ok=True)
        test_folder.mkdir(parents=True, exist_ok=True)

        # Load datasets
        logger.info(f"Loading dataset for sender_id {sender_id}...")
        datasets = AutoencoderDataset.from_disk_data_split(
            sender_id, split_sizes=split_sizes, remove_users_below_n_opens=1
        )
        train, val = datasets[0], datasets[1]
        logger.info(
            f"Dataset loaded. Train size: {len(train.mails)}, Val size: {len(val.mails)}"
        )

        # Generate parameter combinations
        if n_samples is None:
            # Use all combinations (grid search)
            import itertools

            param_keys = list(param_grid.keys())
            param_combinations = list(itertools.product(*param_grid.values()))
            grid = [dict(zip(param_keys, combo)) for combo in param_combinations]
        else:
            # Use random sampling
            grid = list(
                ParameterSampler(
                    param_grid, n_iter=n_samples, random_state=random_state
                )
            )

        logger.info(f"Evaluating {len(grid)} parameter combinations...")

        # Save experiment metadata
        metadata = {
            "sender_id": sender_id,
            "model_name": cls.model_name(),
            "version": version,
            "split_sizes": split_sizes,
            "param_grid": param_grid,
            "n_samples": n_samples,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "total_combinations": len(grid),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(results_folder / "grid_search_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # Run a parallel grid search
        t0 = time.time()
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(cls._run_single_experiment)(
                cfg_dict, exp_number, train, val, results_folder
            )
            for exp_number, cfg_dict in enumerate(grid)
        )
        elapsed_time = time.time() - t0
        logger.info(f"Grid search completed in {elapsed_time:.1f} seconds")

        # Process and save results
        cls._save_grid_search_results(
            results, results_folder, test_folder, elapsed_time
        )

    @classmethod
    def _run_single_experiment(
        cls,
        cfg_dict: dict[str, Any],
        exp_number: int,
        train: AutoencoderDataset,
        val: AutoencoderDataset,
        results_folder: Path,
    ) -> dict[str, Any]:

        exp_folder = results_folder / f"exp_{exp_number}"
        exp_folder.mkdir(exist_ok=True, parents=True)

        with open(exp_folder / "config.json", "w") as f:
            json.dump(cfg_dict, f, indent=4)

        config = cls.config_class()(**cfg_dict)
        model = cls(config=config)

        model.fit(train, val)
        metrics = model.predict(val)

        score = AbstractContextualModel.calculate_auc(metrics)
        _, recalls, precisions = cls.calculate_recall_precision_at_quantiles(metrics)

        results_summary = {
            "experiment_number": exp_number,
            "score": float(score),
            "recalls": [float(r) for r in recalls] if recalls else [],
            "precisions": [float(p) for p in precisions] if precisions else [],
            "config": cfg_dict,
            "n_metrics": len(metrics),
        }

        with open(exp_folder / "results_summary.json", "w") as f:
            json.dump(results_summary, f, indent=4)

        return results_summary

    @classmethod
    def _save_grid_search_results(
        cls,
        results: list[dict[str, Any]],
        results_folder: Path,
        test_folder: Path,
        elapsed_time: float,
    ) -> None:

        results_data = []
        for result in results:
            row = {
                "experiment": result["experiment_number"],
                "score": result["score"],
                **result["config"],
            }

            for i, recall in enumerate(result["recalls"]):
                row[f"recall_{i}"] = recall

            for i, precision in enumerate(result["precisions"]):
                row[f"precision_{i}"] = precision

            row["n_metrics"] = result.get("n_metrics", 0)
            results_data.append(row)

        results_df = pd.DataFrame(results_data)
        results_df.to_csv(results_folder / "all_results.csv", index=False)

        top_k = min(10, len(results))
        top_results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

        summary = {
            "total_experiments": len(results),
            "elapsed_time_seconds": elapsed_time,
            "best_score": top_results[0]["score"] if top_results else 0.0,
            "best_config": top_results[0]["config"] if top_results else {},
            "top_results": top_results,
        }

        with open(results_folder / "grid_search_summary.json", "w") as f:
            json.dump(summary, f, indent=4)

        with open(test_folder / "config.json", "w") as f:
            json.dump(top_results[0]["config"], f, indent=4)

    @staticmethod
    def calculate_recall_precision_at_quantiles(
        metrics: list[ContextualBanditMetrics],
        quantiles: Sequence[float] = (0.25, 0.5, 0.75, 0.85),
    ) -> tuple[list[float], list[float], list[float]]:

        by_mailshot = defaultdict(list)
        for m in metrics:
            if hasattr(m, "present_in_prediction") and not m.present_in_prediction:
                continue
            by_mailshot[m.mailshot_id].append(m)

        q = np.asarray(quantiles, dtype=np.float64)
        n_q = q.size
        rec_sum = np.zeros(n_q, dtype=np.float64)
        prec_sum = np.zeros(n_q, dtype=np.float64)
        weights = np.zeros(n_q, dtype=np.float64)

        for ms in by_mailshot.values():
            y_true = np.fromiter((m.opened for m in ms), dtype=np.int8)
            y_pred = np.fromiter(
                (
                    (m.prediction - m.stage) if hasattr(m, "stage") else m.prediction
                    for m in ms
                ),
                dtype=np.float64,
            )
            pos = y_true.sum()
            if pos == 0:
                continue

            order = np.argsort(y_pred)[::-1]
            y_true_sorted = y_true[order]
            cum_tp = np.cumsum(y_true_sorted, dtype=np.int64)

            k = np.ceil((1.0 - q) * len(y_pred)).astype(int) - 1
            k = np.clip(k, 0, len(y_pred) - 1)

            tp_k = cum_tp[k]
            rec_k = tp_k / pos
            pre_k = tp_k / (k + 1)

            w = len(ms)
            rec_sum += rec_k * w
            prec_sum += pre_k * w
            weights += w

        nz = weights > 0
        final_rec = np.zeros_like(rec_sum)
        final_prec = np.zeros_like(prec_sum)
        final_rec[nz] = rec_sum[nz] / weights[nz]
        final_prec[nz] = prec_sum[nz] / weights[nz]

        sent_pct = (1.0 - q).tolist()
        return sent_pct, final_rec.tolist(), final_prec.tolist()

    @staticmethod
    def calculate_auc(metrics: list[ContextualBanditMetrics]) -> float:
        by_mailshot = defaultdict(list)
        for m in metrics:
            if not m.present_in_prediction:
                continue
            by_mailshot[m.mailshot_id].append(m)

        area_sum = 0.0
        weight_total = 0

        for ms in by_mailshot.values():
            n = len(ms)
            y_true = np.fromiter((m.opened for m in ms), dtype=np.int8, count=n)
            pos = int(y_true.sum())
            if pos == 0:
                continue

            y_pred = np.fromiter(
                (m.prediction - m.stage for m in ms),
                dtype=np.float64,
                count=n,
            )

            order = np.argsort(y_pred)[::-1]
            cum_tp = np.cumsum(y_true[order], dtype=np.int64)

            area_sum += cum_tp.sum() / pos
            weight_total += n

        return area_sum / weight_total if weight_total else 0.0
