import json
import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Self, Set, Tuple, Type

import numpy as np
import torch
from matplotlib import pyplot as plt

from email_outreach.experiment_utils.experiment_constants import (
    CONTEXTUAL_BANDIT_FOLDER_NAME,
)
from email_outreach.ml.shallow_autoencoder.abstract_contextual_model import (
    AbstractConfig,
    AbstractContextualModel,
    ContextualBanditMetrics,
)
from email_outreach.ml.shallow_autoencoder.dataset.autoencoder_dataset import (
    AutoencoderDataset,
)
from email_outreach.ml.shallow_autoencoder.model.autoencoder_chunked import (
    ShallowAutoencoder,
    TrainingMetrics,
)

from email_outreach.ml.shallow_autoencoder.alpha_scheduler import AbstractAlphaScheduler, AlphaSchedulerFactory, \
    AlphaSchedulerArgs
from email_outreach.ml.shallow_autoencoder.dataset.template_weight import AbstractTemplateWeight, TemplateWeightFactory

logger = logging.getLogger(__name__)


@dataclass
class ContextualBanditWithAutoencoderConfig(AbstractConfig):
    d: int = 16
    epochs: int = 30
    lr: float = 0.01
    wd: float = 0.0001
    batch_size: int = 16
    positive_weight: float = 100.0
    alpha_type: str = "constant"
    alpha_params: dict[str, float] = field(default_factory=lambda: {"alpha": 0.1})
    template_weight: str = "uniform"
    template_weight_params: dict[str, float] = field(default_factory=lambda: {})
    G: float = 10000.0
    layer_norm: bool = True
    naive_f: bool = True
    T: int = 2880  # Time in minutes
    num_splits: int = 24  # Number of splits -> K
    sent_by_T: float = 0.15  # Sent by T of all users
    sent_after_T: float = 0.1
    j: int = 0  # Repetition of the experiment
    dropout: float = 0.0
    flipped: bool = False

    def to_dict(self):
        return {
            "d": self.d,
            "epochs": self.epochs,
            "lr": self.lr,
            "wd": self.wd,
            "batch_size": self.batch_size,
            "positive_weight": self.positive_weight,
            "alpha_type": self.alpha_type,
            "alpha_params": self.alpha_params,
            "template_weight": self.template_weight,
            "template_weight_params": self.template_weight_params,
            "G": self.G,
            "layer_norm": self.layer_norm,
            "naive_f": self.naive_f,
            "num_splits": self.num_splits,
            "T": self.T,
            "sent_by_T": self.sent_by_T,
            "sent_after_T": self.sent_after_T,
            "dropout": self.dropout,
            "flipped": self.flipped,
            "j": self.j,
        }

    @classmethod
    def from_json_file(cls, folder: Path, filename: str = "config.json") -> Self:
        with open(folder / filename, "r") as f:
            data = json.load(f)

        return cls(**data)


class ContextualBanditWithAutoencoder(AbstractContextualModel):

    def __init__(self, config: ContextualBanditWithAutoencoderConfig):
        super().__init__(config)
        self._autoencoder: Optional[ShallowAutoencoder] = None
        self.last_p: torch.Tensor = torch.zeros(0)
        self.last_prenormalized_f: torch.Tensor = torch.zeros(0)

        self.alpha_scheduler: AbstractAlphaScheduler = AlphaSchedulerFactory.from_config(config.to_dict())
        self._template_weight: AbstractTemplateWeight | None = None

        self.config = config
        self.calculated_metrics: List[ContextualBanditMetrics] = []
        self.graph_for_n_runs: int = 0
        self.current_mailshot: int = -1
        self.show_plots = False

        self.pjs_predictions: List[np.array] = []
        self.fjs_predictions: List[np.array] = []
        self.sjs_predictions: List[np.array] = []
        self.user_mask: List[np.array] = []

        self._popularity_tensor = None

    @classmethod
    def model_name(cls) -> str:
        return CONTEXTUAL_BANDIT_FOLDER_NAME

    @classmethod
    def config_class(cls) -> Type[ContextualBanditWithAutoencoderConfig]:
        return ContextualBanditWithAutoencoderConfig

    @classmethod
    def metrics_class(cls) -> Type[ContextualBanditMetrics]:
        return ContextualBanditMetrics

    @property
    def autoencoder(self) -> ShallowAutoencoder:
        if self._autoencoder is None:
            raise ValueError("Autoencoder not fitted yet")
        return self._autoencoder

    def fit(
            self,
            train: AutoencoderDataset | torch.Tensor,
            val: AutoencoderDataset | torch.Tensor,
    ) -> None:
        super().fit(train, val)

        weight_config = self.config.to_dict() | {"min_template_id": self.train.min_template_id, "max_template_id": self.train.max_template_id}
        self._template_weight = TemplateWeightFactory.from_config(weight_config)
        self._autoencoder = ShallowAutoencoder(
            n=train.num_users,
            d=self.config.d,
            layer_norm=self.config.layer_norm,
            dropout_p=self.config.dropout,
        )
        self.last_p: torch.Tensor = torch.zeros(train.num_users)
        self.last_prenormalized_f: torch.Tensor = torch.zeros(train.num_users)

        logger.info("Fitting the autoencoder")
        self.autoencoder.fit(
            self.train,
            epochs=self.config.epochs,
            lr=self.config.lr,
            batch_size=self.config.batch_size,
            weight_decay=self.config.wd,
            positive_weight=self.config.positive_weight,
            val=self.val,
            full_training=True,
        )
        # TrainingMetrics.plot_average_ndcg(self.autoencoder.training_metrics)
        # TrainingMetrics.plot_f1_score(self.autoencoder.training_metrics)
        if self.show_plots:
            TrainingMetrics.plot_train_and_val_loss(self.autoencoder.training_metrics)
            TrainingMetrics.plot_average_train_ndcg(self.autoencoder.training_metrics)
            TrainingMetrics.plot_val_ndcg(self.autoencoder.training_metrics)
            TrainingMetrics.plot_val_f1_score(self.autoencoder.training_metrics)
        logger.info("Autoencoder fitted")

    def predict(self, data: AutoencoderDataset) -> List[ContextualBanditMetrics]:
        all_metrics: List[ContextualBanditMetrics] = []
        input_dim = self.train.num_users
        T: int = self.config.T
        num_splits: int = self.config.num_splits  # number of splits
        sent_by_T: float = self.config.sent_by_T
        time_frame_minutes: int = int(T / num_splits)
        initial_batch_size: int = int(self.train.num_users * sent_by_T / num_splits)
        # logger.info(f"config: {self.config}")

        for mailshot_index, mailshot_id in enumerate(data.mailshot_ids):
            self.current_mailshot = mailshot_id
            # logger.info(f"-----------------------------------")
            # logger.info(f"Predicting for mailshot {mailshot_id}")
            already_sent: List[int] = []  # List of already sent users
            already_scored: List[float] = []  # List of scores for already sent users
            opened: torch.tensor = torch.zeros(input_dim)
            mailshot_users: List[int] = data.mailshot_user_indices(mailshot_id)

            # User's index: stage
            sent_at_stage: Dict[int, int] = {}
            all_current_opens: Set[int] = set()
            newly_opened: Set[int] = set()

            for batch_i in range(num_splits):
                # logger.info(f"Predicting batch {i}")
                predicted_batch, predicted_scores = self.predict_batch(
                    mailshot_users,
                    already_sent,
                    opened,
                    initial_batch_size,
                    newly_opened,
                    scheduler_args= AlphaSchedulerArgs(batch=batch_i, open_frac=len(all_current_opens)/input_dim),
                )
                already_sent.extend(predicted_batch)
                already_scored.extend(predicted_scores)
                sent_at_stage.update({user: batch_i for user in predicted_batch})
                opens_from_batch: Set[int] = set(
                    self.update_opens(
                        data, mailshot_id, sent_at_stage, time_frame_minutes
                    )
                )
                newly_opened = opens_from_batch - all_current_opens
                # print(newly_opened)
                all_current_opens.update(opens_from_batch)
                opened[list(opens_from_batch)] = 1
                current_opens = len(opens_from_batch)

            # Final prediction and metrics
            predictions: torch.Tensor = self.predict_final_send(
                mailshot_users, already_sent, opened, newly_opened, scheduler_args = AlphaSchedulerArgs(batch=num_splits, open_frac=len(all_current_opens)/input_dim)
            )
            metrics: List[ContextualBanditMetrics] = self.final_metrics(
                mailshot_index, data, mailshot_users, already_sent, predictions
            )
            for m in metrics:
                m.stage = num_splits

            # Update metrics for already sent users (in the initial stages)
            for user, stage in sent_at_stage.items():
                metrics[user].stage = stage
            for user, score in zip(already_sent, already_scored):
                metrics[user].prediction = score

            all_metrics.extend(metrics)
        self.calculated_metrics = all_metrics
        return all_metrics

    @staticmethod
    def update_opens(
            dataset: AutoencoderDataset,
            mailshot_id: int,
            sent_at_stage: Dict[int, int],
            frame_minutes: int,
    ) -> List[int]:
        opened_indices: List[int] = []
        grouped_sent_at_stage = defaultdict(list)
        # logger.info(f"Updating opens for mailshot {mailshot_id}, keys: {set(sent_at_stage.values())}")
        max_stage = max(sent_at_stage.values())
        for user, stage in sent_at_stage.items():
            grouped_sent_at_stage[stage].append(user)

        for stage, users in grouped_sent_at_stage.items():
            opened_indices.extend(
                dataset.select_opened_indices(
                    mailshot_id, users, frame_minutes * (max_stage - stage + 1)
                )
            )

        return opened_indices

    def calculate_scores(
            self,
            user_indices: List[int],
            sent_indices: List[int],
            opened: torch.tensor,
            newly_opened: Set[int],
            scheduler_args: AlphaSchedulerArgs
    ) -> torch.Tensor:
        # Calculate p, f, s
        fs: torch.tensor = self.calculate_f(opened, newly_opened) if self.config.naive_f else self.non_naive_f(opened)
        if self.last_p.sum() == 0:
            ps: torch.tensor = self.calculate_p()
            self.last_p = ps
        else:
            ps: torch.tensor = self.last_p
        popularity: torch.tensor = self.calculate_popularity()
        ps = ps * popularity
        s: torch.tensor = self.calculate_s(ps, fs, scheduler_args)

        # Sample from Beta distribution
        samples: torch.tensor = self.calculate_samples(s, self.config.G)

        # Don't consider already sent samples
        samples[sent_indices] = 0

        # Don't consider users not in user_indices
        user_mask = self.user_indices_to_mask(user_indices)
        samples[~user_mask] = 0

        self.pjs_predictions.append(ps.detach().numpy())
        self.fjs_predictions.append(fs.detach().numpy())
        self.sjs_predictions.append(s.detach().numpy())
        self.user_mask.append(user_mask.detach().numpy())
        return samples

    def predict_batch(
            self,
            user_indices: List[int],
            sent_indices: List[int],
            opened: torch.tensor,
            n_best: int,
            newly_opened: Set[int],
            scheduler_args: AlphaSchedulerArgs
    ) -> Tuple[List[int], List[float]]:
        # Calculate p, f, s
        samples = self.calculate_scores(
            user_indices, sent_indices, opened, newly_opened, scheduler_args
        )

        # Select indices of top N samples
        top_indices: List[int] = torch.topk(samples, n_best).indices.tolist()
        top_scores: List[float] = samples[top_indices].tolist()
        return top_indices, top_scores  # Return top N indices as list and their scores

    def user_indices_to_mask(self, user_indices: List[int]) -> torch.tensor:
        mask = torch.zeros(self.train.num_users, dtype=torch.bool)
        mask[torch.tensor(user_indices)] = True
        return mask

    def predict_final_send(
            self,
            user_indices: List[int],
            sent_indices: List[int],
            opened: torch.tensor,
            newly_opened: Set[int],
            scheduler_args: AlphaSchedulerArgs
    ) -> torch.Tensor:
        samples = self.calculate_scores(
            user_indices, sent_indices, opened, newly_opened, scheduler_args
        )
        return samples

    def final_metrics(
            self,
            mailshot_id: int,
            dataset: AutoencoderDataset,
            user_indices: List[int],
            sent_indices: List[int],
            final_predictions: torch.Tensor,
    ) -> List[ContextualBanditMetrics]:
        binary_results: torch.Tensor = dataset[mailshot_id][
            2
        ]  # 1 because the 0 is masked for training
        prediction = deepcopy(final_predictions)

        # Put 1s for sent_indices
        prediction[sent_indices] = 1

        # User indices mask
        user_mask = self.user_indices_to_mask(user_indices)

        # User cluster tensor
        # user_clusters = self.train.user_clusters.sort_values('user_id')['cluster'].tolist()
        user_clusters = [0] * self.train.num_users

        # Calculate metrics
        metrics: List[ContextualBanditMetrics] = ContextualBanditMetrics.from_lists(
            mailshot_id,
            user_mask.tolist(),
            binary_results.tolist(),
            prediction.tolist(),
            user_clusters,
        )
        return metrics

    def calculate_popularity(self) -> torch.tensor:
        if not self._template_weight:
            raise ValueError("No template weight")

        if self._popularity_tensor is None:
            self._popularity_tensor = self.train.popularity_tensor(self._template_weight)
        return self._popularity_tensor

    def calculate_p(self) -> torch.tensor:
        # logger.info("Calculating p")
        batch_size = 100
        input_dim = self.train.num_users
        all_averages = []
        all_variances = []
        for start in range(0, input_dim, batch_size):
            end = min(start + batch_size, input_dim)
            users = torch.zeros(end - start, input_dim)
            for i in range(start, end):
                users[i - start, i] = 1  # train.average_opens
            if self.config.layer_norm:
                preds = self.autoencoder.predict(users)
            else:
                preds = self.autoencoder.predict_for_user(users)

            # Remove self-contribution by zeroing the diagonal
            no_self_users = preds.clone()
            for i in range(start, end):
                no_self_users[i - start, i] = 0

            # Calculate the average
            averages = no_self_users.mean(dim=1)
            variances = no_self_users.var(dim=1)
            all_averages.append(averages)
            all_variances.append(variances)

        all_averages = torch.cat(all_averages)
        # Show only values of true opens
        if self.show_plots:
            plt.hist(all_averages.detach().numpy().flatten(), bins=100, color="blue")
            true_opens: torch.Tensor = self.val.tensor_of_opens_of_mailshot(
                self.current_mailshot
            )
            averages_of_trues = all_averages[true_opens == 1]
            plt.hist(
                averages_of_trues.detach().numpy().flatten(), bins=100, color="orange"
            )
            plt.title("Histogram of p")
            plt.show()
        # Fill the averages to the input_dim size
        # return torch.ones(all_averages.shape) - all_averages
        return all_averages

    def calculate_f(
            self, results: torch.Tensor, newly_opened: Set[int]
    ) -> torch.Tensor:
        # logger.info(f"Calculating f Original")
        if results.sum() == 0:
            self.last_prenormalized_f = torch.zeros_like(results)
            return torch.zeros_like(results)

        # Indices of the results
        f = torch.zeros_like(results, dtype=torch.float)
        for index in newly_opened:
            user = torch.zeros_like(results)
            user[index] = 1
            if self.config.layer_norm:
                current_f = self.autoencoder.predict(user.unsqueeze(0)).squeeze(0)
            else:
                current_f = self.autoencoder.predict_for_user(
                    user.unsqueeze(0)
                ).squeeze(0)
            assert current_f.max() <= 1, f"Max f: {current_f.max()}, should be <=1"
            f += current_f

        self.last_prenormalized_f += f
        normalized_f = self.last_prenormalized_f / sum(results)
        if self.config.flipped:
            return torch.ones(normalized_f.shape) - normalized_f
        else:
            return normalized_f

    def non_naive_f(self, results: torch.Tensor) -> torch.Tensor:
        current_f = self.autoencoder.predict(results)
        if self.config.flipped:
            return torch.ones(current_f.shape) - current_f
        else:
            return current_f

    def calculate_f_ff(
            self, results: torch.Tensor, newly_opened: Set[int]
    ) -> torch.Tensor:
        logger.info(f"Calculating f FF")
        if results.sum() == 0:
            return torch.zeros_like(results)

        current_f = self.autoencoder.predict(results)
        if self.graph_for_n_runs > 0:
            with torch.no_grad():
                plt.hist(current_f.detach().numpy().flatten(), bins=100, color="blue")
                plt.title("Histogram of f")
                # Show only values of true opens
                true_opens: torch.Tensor = self.val.tensor_of_opens_of_mailshot(
                    self.current_mailshot
                )
                averages_of_trues = current_f[true_opens == 1]
                plt.hist(
                    averages_of_trues.detach().numpy().flatten(),
                    bins=100,
                    color="orange",
                )
                self.graph_for_n_runs -= 1
                plt.show()
        if self.config.flipped:
            return torch.ones(current_f.shape) - current_f
        else:
            return current_f

    def calculate_s(self, p: torch.Tensor, f: torch.Tensor, scheduler_args: AlphaSchedulerArgs) -> torch.Tensor:
        # 𝑠𝑗 = 𝛼*𝑝_𝑗 + (1 − 𝛼)*𝑓_𝑗(𝑡)
        # logger.info("Calculating s")
        s = self.alpha_scheduler(scheduler_args) * p + (1 - self.alpha_scheduler(scheduler_args)) * f
        if f.max() == 1:
            print("Max f: 1")
            return f
        assert s.max() <= 1, f"Max s: {s.max()}, should be <=1"
        return s

    @staticmethod
    def calculate_samples(s: torch.Tensor, G: float) -> torch.Tensor:
        # logger.info("Calculating samples")
        alphas = torch.clamp(s * G, min=1e-12)
        betas = torch.clamp((1 - s) * G, min=1e-12)
        samples = torch.distributions.Beta(alphas, betas).sample()
        return samples
