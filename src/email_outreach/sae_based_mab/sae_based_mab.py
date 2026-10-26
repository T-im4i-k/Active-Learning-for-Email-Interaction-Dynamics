import logging
from collections import defaultdict
from copy import deepcopy
from functools import cached_property

import numpy as np
import torch

from email_outreach.sae_based_mab.abstract_mab import AbstractContextualModel
from email_outreach.sae_based_mab.coef_scheduler import (AbstractCoefScheduler,
                                                         AlphaSchedulerFactory,
                                                         BetaSchedulerFactory,
                                                         CoefSchedulerArgs)
from email_outreach.sae_based_mab.config import (
    ContextualBanditMetrics, ContextualBanditWithAutoencoderConfig)
from email_outreach.sae_based_mab.dataset import (AutoencoderDataset,
                                                  NegativeAutoencoderDataset,
                                                  PositiveAutoencoderDataset)
from email_outreach.sae_based_mab.sae import ShallowAutoencoder
from email_outreach.sae_based_mab.template_weight import (
    AbstractTemplateWeight, TemplateWeightFactory)
from email_outreach.utils.constants import ROOT_FOLDER

logger = logging.getLogger(__name__)


class ContextualBanditWithAutoencoder(AbstractContextualModel):

    def __init__(self, config: ContextualBanditWithAutoencoderConfig):
        super().__init__(config)

        self.config = config
        self._autoencoder: ShallowAutoencoder | None = None
        self._negative_autoencoder: ShallowAutoencoder | None = None

        self.alpha_scheduler: AbstractCoefScheduler = AlphaSchedulerFactory.from_config(
            config.to_dict()
        )
        self.beta_scheduler: AbstractCoefScheduler | None = (
            BetaSchedulerFactory.from_config(config.to_dict())
        )
        self._template_weight: AbstractTemplateWeight | None = None

        self.calculated_metrics: list[ContextualBanditMetrics] = []
        self.sent_at_stage: dict[int, int] = {}
        self.user_mask: list[np.ndarray] = []

        self._naive_f_partial: torch.Tensor | None = None

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

        super().fit(train, val)

        weight_config = self.config.to_dict() | {
            "min_template_id": train.min_template_id,
            "max_template_id": train.max_template_id,
        }
        self._template_weight = TemplateWeightFactory.from_config(weight_config)

        self._autoencoder = ShallowAutoencoder(
            n=train.num_users, d=self.config.d, dropout_p=self.config.dropout
        )

        self._naive_f_partial: torch.Tensor = torch.zeros(train.num_users)

        logger.info("Fitting the autoencoder")

        if self.config.double_sae:
            self._negative_autoencoder = ShallowAutoencoder(
                n=train.num_users,
                d=self.config.d,
                dropout_p=self.config.dropout,
            )

            self._negative_autoencoder.fit(
                NegativeAutoencoderDataset(train, noise=self.config.noise),
                val=NegativeAutoencoderDataset(val),
                epochs=self.config.epochs,
                lr=self.config.lr,
                batch_size=self.config.batch_size,
                weight_decay=self.config.wd,
                positive_weight=self.config.negative_weight,
            )

            # np.save(ROOT_FOLDER / "additional_data" / "D" / "neg_sae_D.npy", self._negative_autoencoder.D.detach().numpy())

        self.autoencoder.fit(
            PositiveAutoencoderDataset(train, noise=self.config.noise),
            val=PositiveAutoencoderDataset(val),
            epochs=self.config.epochs,
            lr=self.config.lr,
            batch_size=self.config.batch_size,
            weight_decay=self.config.wd,
            positive_weight=self.config.positive_weight,
        )

        # np.save(ROOT_FOLDER / "additional_data" / "D" /"sae_D.npy", self.autoencoder.D.detach().numpy())
        logger.info("Autoencoder fitted")

    def predict(self, data: AutoencoderDataset) -> list[ContextualBanditMetrics]:
        all_metrics: list[ContextualBanditMetrics] = []

        for mailshot_index, mailshot_id in enumerate(data.mailshot_ids):
            already_sent: list[int] = []
            already_scored: list[float] = []
            opened: torch.tensor = torch.zeros(self._train.num_users)
            mailshot_users: list[int] = data.mailshot_user_indices(mailshot_id)

            all_current_opens: set[int] = set()
            newly_opened: set[int] = set()

            for batch_i in range(self.config.num_splits):
                predicted_batch, predicted_scores = self.predict_batch(
                    mailshot_users,
                    already_sent,
                    opened,
                    int(self._train.num_users * self.config.sent_by_T / self.config.num_splits),
                    newly_opened,
                    scheduler_args=CoefSchedulerArgs(
                        batch=batch_i, open_frac=len(all_current_opens) / self._train.num_users
                    ),
                )
                already_sent.extend(predicted_batch)
                already_scored.extend(predicted_scores)
                self.sent_at_stage.update({user: batch_i for user in predicted_batch})
                opens_from_batch: set[int] = set(
                    self.update_opens(
                        data, mailshot_id, self.sent_at_stage, int(self.config.T / self.config.num_splits)
                    )
                )
                newly_opened = opens_from_batch - all_current_opens
                all_current_opens.update(opens_from_batch)
                opened[list(opens_from_batch)] = 1

            predictions: torch.Tensor = self.predict_final_send(
                mailshot_users,
                already_sent,
                opened,
                newly_opened,
                scheduler_args=CoefSchedulerArgs(
                    batch=self.config.num_splits, open_frac=len(all_current_opens) / self._train.num_users
                ),
            )
            metrics: list[ContextualBanditMetrics] = self.final_metrics(
                mailshot_index, data, mailshot_users, already_sent, predictions
            )
            for m in metrics:
                m.stage = self.config.num_splits

            for user, stage in self.sent_at_stage.items():
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
        sent_at_stage: dict[int, int],
        frame_minutes: int,
    ) -> list[int]:
        opened_indices: list[int] = []
        grouped_sent_at_stage = defaultdict(list)
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

    # ===== p =====
    @cached_property
    def p_score(self) -> torch.Tensor:
        batch_size = 256
        input_dim = self._train.num_users

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
        all_variances = torch.cat(all_variances)

        match self.config.p_type:
            case "mean":
                return all_averages
            case "var":
                return 4 * all_variances
            case "std":
                return 2 * torch.sqrt(all_variances)
            case _:
                raise ValueError("Unknown p value")

    # ===== f =====

    def calculate_f(
        self, results: torch.Tensor, newly_opened: set[int]
    ) -> torch.Tensor:
        if results.sum() == 0:
            self._naive_f_partial = torch.zeros_like(results)
            return torch.zeros_like(results)

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

        self._naive_f_partial += f
        normalized_f = self._naive_f_partial / sum(results)

        return normalized_f

    def non_naive_f(
        self, results: torch.Tensor, sent_indices: list[int]
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

    def get_f(self, opened, newly_opened, sent_indices: list[int]) -> torch.Tensor:
        match self.config.naive_f:
            case True:
                return self.calculate_f(opened, newly_opened)
            case False:
                return self.non_naive_f(opened, sent_indices)

    # ===== s =====

    def calculate_s(
        self,
        phi: torch.Tensor,
        p: torch.Tensor,
        f: torch.Tensor,
        scheduler_args: CoefSchedulerArgs,
    ) -> torch.Tensor:
        s = (
            self.alpha_scheduler(scheduler_args) * phi * p
            + (1 - self.alpha_scheduler(scheduler_args)) * f
        )
        return s

    def calculate_s_revised(self, phi, p, f, scheduler_args: CoefSchedulerArgs):
        alpha = self.alpha_scheduler(scheduler_args)
        beta = self.beta_scheduler(scheduler_args)

        pi = alpha * phi + (1 - alpha) * f
        u = beta * p + (1 - beta)
        s = pi * u

        return s

    def get_s(
        self,
        phi: torch.Tensor,
        p: torch.Tensor,
        f: torch.Tensor,
        scheduler_args: CoefSchedulerArgs,
    ) -> torch.Tensor:
        match self.config.revised_s:
            case False:
                return self.calculate_s(phi, p, f, scheduler_args)
            case True:
                return self.calculate_s_revised(phi, p, f, scheduler_args)

    # ===== phi =====
    @cached_property
    def phi_score(self) -> torch.Tensor:
        return self._train.popularity_tensor(self._template_weight)

    # ===== TS MAB =====

    def calculate_scores(
        self,
        user_indices: list[int],
        sent_indices: list[int],
        opened: torch.Tensor,
        newly_opened: set[int],
        scheduler_args: CoefSchedulerArgs,
    ) -> torch.Tensor:

        f: torch.Tensor = self.get_f(opened, newly_opened, sent_indices)
        p: torch.Tensor = self.p_score
        phi: torch.Tensor = self.phi_score

        s: torch.Tensor = self.get_s(phi, p, f, scheduler_args)

        samples: torch.tensor = self.calculate_samples(s, self.config.G)
        samples[sent_indices] = 0

        user_mask = self.user_indices_to_mask(user_indices)
        samples[~user_mask] = 0
        self.user_mask.append(user_mask.detach().numpy())

        return samples

    def predict_batch(
        self,
        user_indices: list[int],
        sent_indices: list[int],
        opened: torch.tensor,
        n_best: int,
        newly_opened: set[int],
        scheduler_args: CoefSchedulerArgs,
    ) -> tuple[list[int], list[float]]:
        samples = self.calculate_scores(
            user_indices, sent_indices, opened, newly_opened, scheduler_args
        )

        top_indices: list[int] = torch.topk(samples, n_best).indices.tolist()
        top_scores: list[float] = samples[top_indices].tolist()
        return top_indices, top_scores

    def user_indices_to_mask(self, user_indices: list[int]) -> torch.tensor:
        mask = torch.zeros(self._train.num_users, dtype=torch.bool)
        mask[torch.tensor(user_indices)] = True
        return mask

    def predict_final_send(
        self,
        user_indices: list[int],
        sent_indices: list[int],
        opened: torch.tensor,
        newly_opened: set[int],
        scheduler_args: CoefSchedulerArgs,
    ) -> torch.Tensor:
        return self.calculate_scores(
            user_indices, sent_indices, opened, newly_opened, scheduler_args
        )

    def final_metrics(
        self,
        mailshot_id: int,
        dataset: AutoencoderDataset,
        user_indices: list[int],
        sent_indices: list[int],
        final_predictions: torch.Tensor,
    ) -> list[ContextualBanditMetrics]:
        binary_results: torch.Tensor = dataset[mailshot_id]
        prediction = deepcopy(final_predictions)
        prediction[sent_indices] = 1
        user_mask = self.user_indices_to_mask(user_indices)

        return ContextualBanditMetrics.from_lists(
            mailshot_id,
            user_mask.tolist(),
            binary_results.tolist(),
            prediction.tolist(),
        )

    @staticmethod
    def calculate_samples(s: torch.Tensor, G: float) -> torch.Tensor:
        alphas = torch.clamp(s * G, min=1e-12)
        betas = torch.clamp((1 - s) * G, min=1e-12)
        samples = torch.distributions.Beta(alphas, betas).sample()
        return samples


# ===== Additional =====


def max_p_sigma_rows(
    autoencoder: ShallowAutoencoder,
    all_averages: torch.Tensor,
    all_variances: torch.Tensor,
) -> None:
    max_average_idx = torch.argmax(all_averages)
    max_average_input = torch.zeros(autoencoder.n)
    max_average_input[max_average_idx] = 1

    max_average_row = autoencoder.predict(max_average_input)
    np.save(
        ROOT_FOLDER / "additional_data" / "max_p_sigma_row" / "avg.npy",
        np.array(max_average_row.detach().numpy(), dtype=np.float64),
    )

    max_var_idx = torch.argmax(all_variances)
    max_var_input = torch.zeros(autoencoder.n)
    max_var_input[max_var_idx] = 1

    max_var_row = autoencoder.predict(max_var_input)
    np.save(
        ROOT_FOLDER / "additional_data" / "max_p_sigma_row" / "var.npy",
        np.array(max_var_row.detach().numpy(), dtype=np.float64),
    )


def max_p_rank_flip_sigma_row(
    autoencoder: ShallowAutoencoder,
    all_averages: torch.Tensor,
    all_variances: torch.Tensor,
):
    from scipy.stats import rankdata

    mean_ranking = rankdata(-all_averages, method="min")
    var_ranking = rankdata(-all_variances, method="min")

    rank_flips = torch.abs(torch.tensor(var_ranking - mean_ranking))

    biggest_rank_flip = torch.argmax(rank_flips)
    biggest_rank_flip_vec = torch.zeros(autoencoder.n)
    biggest_rank_flip_vec[biggest_rank_flip] = 1

    sigma_row = autoencoder.predict(biggest_rank_flip_vec.unsqueeze(0))
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
