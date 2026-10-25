import logging
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Set, Tuple, Type

import numpy as np
import torch

from email_outreach.experiment_utils.experiment_constants import (
    CONTEXTUAL_BANDIT_FOLDER_NAME,
)
from email_outreach.ml.shallow_autoencoder.abstract_contextual_model import (
    AbstractContextualModel,
)
from email_outreach.ml.shallow_autoencoder.autoencoder_chunked import ShallowAutoencoder
from email_outreach.ml.shallow_autoencoder.coef_scheduler import (
    AbstractCoefScheduler,
    AlphaSchedulerFactory,
    BetaSchedulerFactory,
    CoefSchedulerArgs,
)
from email_outreach.ml.shallow_autoencoder.config import (
    ContextualBanditMetrics,
    ContextualBanditWithAutoencoderConfig,
)
from email_outreach.ml.shallow_autoencoder.dataset.autoencoder_dataset import (
    AutoencoderDataset,
    NegativeAutoencoderDataset,
)
from email_outreach.ml.shallow_autoencoder.dataset.template_weight import (
    AbstractTemplateWeight,
    TemplateWeightFactory,
)
from email_outreach.utils.constants import ROOT_FOLDER

logger = logging.getLogger(__name__)


class ContextualBanditWithAutoencoder(AbstractContextualModel):

    def __init__(self, config: ContextualBanditWithAutoencoderConfig):
        super().__init__(config)
        self._autoencoder = None
        self.last_p: torch.Tensor = torch.zeros(0)
        self.last_prenormalized_f: torch.Tensor = torch.zeros(0)

        self.alpha_scheduler: AbstractCoefScheduler = AlphaSchedulerFactory.from_config(
            config.to_dict()
        )
        self.beta_scheduler: AbstractCoefScheduler | None = (
            BetaSchedulerFactory.from_config(config.to_dict())
        )
        self._template_weight: AbstractTemplateWeight | None = None

        self.config = config
        self.calculated_metrics: List[ContextualBanditMetrics] = []

        self.sent_at_stage: Dict[int, int] = {}

        self.p_predictions: list[np.array] = []

        self.pjs_predictions: List[np.array] = []
        self.fjs_predictions: List[np.array] = []
        self.sjs_predictions: List[np.array] = []

        self.u_predictions: list[np.ndarray] = []
        self.pi_predictions: list[np.ndarray] = []

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
        train: AutoencoderDataset,
        val: AutoencoderDataset,
    ) -> None:

        # np.save(ROOT_FOLDER / "additional_data" / "X" /"train_X.npy", train._mailshot_embeddings)

        super().fit(train, val)

        weight_config = self.config.to_dict() | {
            "min_template_id": self.train.min_template_id,
            "max_template_id": self.train.max_template_id,
        }
        self._template_weight = TemplateWeightFactory.from_config(weight_config)

        self._autoencoder = ShallowAutoencoder(
            n=train.num_users, d=self.config.d, dropout_p=self.config.dropout
        )

        self.last_p: torch.Tensor = torch.zeros(train.num_users)
        self.last_prenormalized_f: torch.Tensor = torch.zeros(train.num_users)

        logger.info("Fitting the autoencoder")

        if self.config.double_sae:
            self._negative_autoencoder = ShallowAutoencoder(
                n=train.num_users,
                d=self.config.d,
                dropout_p=self.config.dropout,
            )

            self._negative_autoencoder.fit(
                NegativeAutoencoderDataset(train),
                epochs=self.config.epochs,
                lr=self.config.lr,
                batch_size=self.config.batch_size,
                weight_decay=self.config.wd,
                positive_weight=self.config.negative_weight,
                val=val,
            )

            # np.save(ROOT_FOLDER / "additional_data" / "D" / "neg_sae_D.npy", self._negative_autoencoder.D.detach().numpy())

        self.autoencoder.fit(
            train,
            epochs=self.config.epochs,
            lr=self.config.lr,
            batch_size=self.config.batch_size,
            weight_decay=self.config.wd,
            positive_weight=self.config.positive_weight,
            val=val,
        )

        # np.save(ROOT_FOLDER / "additional_data" / "D" /"sae_D.npy", self.autoencoder.D.detach().numpy())
        logger.info("Autoencoder fitted")

    def predict(self, data: AutoencoderDataset) -> List[ContextualBanditMetrics]:
        all_metrics: List[ContextualBanditMetrics] = []
        input_dim = self.train.num_users
        T: int = self.config.T
        num_splits: int = self.config.num_splits  # number of splits
        sent_by_T: float = self.config.sent_by_T
        time_frame_minutes: int = int(T / num_splits)
        initial_batch_size: int = int(self.train.num_users * sent_by_T / num_splits)

        for mailshot_index, mailshot_id in enumerate(data.mailshot_ids):
            # logger.info(f"-----------------------------------")
            # logger.info(f"Predicting for mailshot {mailshot_id}")
            already_sent: List[int] = []  # List of already sent users
            already_scored: List[float] = []  # List of scores for already sent users
            opened: torch.tensor = torch.zeros(input_dim)
            mailshot_users: List[int] = data.mailshot_user_indices(mailshot_id)

            # User's index: stage
            sent_at_stage: Dict[int, int] = {}
            self.sent_at_stage = (
                sent_at_stage  # same dict object; .update() below stays in sync
            )
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
                    scheduler_args=CoefSchedulerArgs(
                        batch=batch_i, open_frac=len(all_current_opens) / input_dim
                    ),
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
                mailshot_users,
                already_sent,
                opened,
                newly_opened,
                scheduler_args=CoefSchedulerArgs(
                    batch=num_splits, open_frac=len(all_current_opens) / input_dim
                ),
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
            # np.save(ROOT_FOLDER / "additional_data" / "u_predictions.npy", np.array(self.u_predictions, dtype=np.float64))
            # np.save(ROOT_FOLDER / "additional_data" / "pi_predictions.npy", np.array(self.pi_predictions, dtype=np.float64))
            print("Saving results")

        self.calculated_metrics = all_metrics

        # self.u_predictions = np.array(self.u_predictions)
        # self.pi_predictions = np.array(self.pi_predictions)

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

    def calculate_s_revised(self, phi, p, f, scheduler_args: CoefSchedulerArgs):
        # 𝑠𝑗 = 𝛼*𝑝_𝑗 + (1 − 𝛼)*𝑓_𝑗(𝑡)
        if self.beta_scheduler is None:
            raise ValueError("Beta scheduler has not been initialized")
        alpha = self.alpha_scheduler(scheduler_args)
        beta = self.beta_scheduler(scheduler_args)

        pi = alpha * phi + (1 - alpha) * f
        u = beta * p + (1 - beta)
        s = pi * u

        self.u_predictions.append(u.detach().numpy())
        self.pi_predictions.append(pi.detach().numpy())

        return s

    def calculate_scores(
        self,
        user_indices: List[int],
        sent_indices: List[int],
        opened: torch.tensor,
        newly_opened: Set[int],
        scheduler_args: CoefSchedulerArgs,
    ) -> torch.Tensor:
        # Calculate p, f, s

        match self.config.naive_f:
            case True:
                fs = self.calculate_f(opened, newly_opened)
            case False:
                fs = self.non_naive_f(opened, sent_indices)

        if self.last_p.sum() == 0:
            ps: torch.tensor = self.calculate_p()
            self.last_p = ps

            # np.save(ROOT_FOLDER / "additional_data" / "p_scores" / "p_var.npy",
            #         np.array(self.last_p.detach().numpy(), dtype=np.float64))
        else:
            ps: torch.tensor = self.last_p

        popularity: torch.tensor = self.calculate_popularity()

        p_phis = ps * popularity

        match self.config.revised_s:
            case False:
                s = self.calculate_s(p_phis, fs, scheduler_args)
            case True:
                s = self.calculate_s_revised(popularity, ps, fs, scheduler_args)

        # Sample from Beta distribution
        samples: torch.tensor = self.calculate_samples(s, self.config.G)

        # Don't consider already sent samples
        samples[sent_indices] = 0

        # Don't consider users not in user_indices
        user_mask = self.user_indices_to_mask(user_indices)
        samples[~user_mask] = 0

        self.pjs_predictions.append(p_phis.detach().numpy())
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
        scheduler_args: CoefSchedulerArgs,
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
        scheduler_args: CoefSchedulerArgs,
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
            1
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
            self._popularity_tensor = self.train.popularity_tensor(
                self._template_weight
            )

        return self._popularity_tensor

    def calculate_p(self) -> torch.Tensor:
        batch_size = 256
        input_dim = self.train.num_users

        all_averages = []
        all_variances = []

        for start in range(0, input_dim, batch_size):
            end = min(start + batch_size, input_dim)
            users = torch.zeros(end - start, input_dim)
            for i in range(start, end):
                users[i - start, i] = 1
            no_self_users = self.autoencoder.predict(users)
            for i in range(start, end):
                no_self_users[i - start, i] = 0

            averages = no_self_users.mean(dim=1)
            variances = no_self_users.var(dim=1)

            all_averages.append(averages)
            all_variances.append(variances)

        all_averages = torch.cat(all_averages)

        max_average_idx = torch.argmax(all_averages)
        max_average_input = torch.zeros(input_dim)
        max_average_input[max_average_idx] = 1

        max_average_row = self.autoencoder.predict(max_average_input)
        # np.save(ROOT_FOLDER / "additional_data" / "max_p_sigma_row" / "avg.npy",
        #         np.array(max_average_row.detach().numpy(), dtype=np.float64))

        all_variances = torch.cat(all_variances)

        max_var_idx = torch.argmax(all_variances)
        max_var_input = torch.zeros(input_dim)
        max_var_input[max_var_idx] = 1

        max_var_row = self.autoencoder.predict(max_var_input)
        # np.save(ROOT_FOLDER / "additional_data" / "max_p_sigma_row" / "var.npy",
        #         np.array(max_var_row.detach().numpy(), dtype=np.float64))

        # ==== ===
        from scipy.stats import rankdata

        mean_ranking = rankdata(-all_averages, method="min")
        var_ranking = rankdata(-all_variances, method="min")

        rank_flips = torch.abs(torch.tensor(var_ranking - mean_ranking))

        biggest_rank_flip = torch.argmax(rank_flips)
        biggest_rank_flip_vec = torch.zeros(self.train.num_users)
        biggest_rank_flip_vec[biggest_rank_flip] = 1

        sigma_row = self.autoencoder.predict(biggest_rank_flip_vec.unsqueeze(0))
        np.save(
            ROOT_FOLDER / "additional_data" / "p_scores" / f"rank_flip_sigma_row.npy",
            np.array(sigma_row.detach().numpy(), dtype=np.float64),
        )

        np.save(
            ROOT_FOLDER / "additional_data" / "p_scores" / f"p_mean.npy",
            np.array(all_averages.detach().numpy(), dtype=np.float64),
        )

        np.save(
            ROOT_FOLDER / "additional_data" / "p_scores" / f"p_var.npy",
            np.array(4 * all_variances.detach().numpy(), dtype=np.float64),
        )

        match self.config.p_type:
            case "mean":
                return all_averages
            case "var":
                return 4 * all_variances
            case "std":
                return 2 * torch.sqrt(all_variances)
            case _:
                raise ValueError("Unknown p value")

    def calculate_f(
        self, results: torch.Tensor, newly_opened: Set[int]
    ) -> torch.Tensor:
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
            f += current_f

        self.last_prenormalized_f += f
        normalized_f = self.last_prenormalized_f / sum(results)
        if self.config.flipped:
            return torch.ones(normalized_f.shape) - normalized_f
        else:
            return normalized_f

    def non_naive_f(
        self, results: torch.Tensor, sent_indices: List[int]
    ) -> torch.Tensor:

        if not self.config.double_sae:
            current_f = torch.sigmoid(
                self.autoencoder.predict_logit(results) * self.config.f_confidence
            )
        else:
            sent_tensor = torch.as_tensor(sent_indices, dtype=torch.long)
            unopened = sent_tensor[results[sent_tensor] == 0]
            negative_results = torch.zeros_like(results)
            negative_results[unopened] = 1

            pos_logit = self.autoencoder.predict_logit(results)
            neg_logit = self._negative_autoencoder.predict_logit(negative_results)
            current_f = torch.sigmoid(
                (
                    (1 - self.config.negative_feedback_ratio) * pos_logit
                    - self.config.negative_feedback_ratio * neg_logit
                )
                * self.config.f_confidence
            )

        return current_f

    def calculate_s(
        self, p: torch.Tensor, f: torch.Tensor, scheduler_args: CoefSchedulerArgs
    ) -> torch.Tensor:
        s = (
            self.alpha_scheduler(scheduler_args) * p
            + (1 - self.alpha_scheduler(scheduler_args)) * f
        )
        return s

    @staticmethod
    def calculate_samples(s: torch.Tensor, G: float) -> torch.Tensor:
        # logger.info("Calculating samples")
        alphas = torch.clamp(s * G, min=1e-12)
        betas = torch.clamp((1 - s) * G, min=1e-12)
        samples = torch.distributions.Beta(alphas, betas).sample()
        return samples
