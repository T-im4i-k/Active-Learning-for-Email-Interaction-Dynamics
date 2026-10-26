from __future__ import annotations

import logging
from itertools import accumulate, pairwise
from typing import Self, Sequence
from functools import cached_property

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from email_outreach.ml.shallow_autoencoder.template_weight import (
    AbstractTemplateWeight,
)
from email_outreach.utils.data_loader_utils import load_data_for_sender

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("mailshot_id", "user_id", "opened", "time_to_open")
MIN_MAILS_PER_MAILSHOT = 100

# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def _drop_users_below_min_opens(mails: pd.DataFrame, min_opens: int) -> pd.DataFrame:
    opens_per_user = mails.groupby("user_id")["opened"].sum()
    kept_users = opens_per_user.index[opens_per_user >= min_opens]
    return mails[mails["user_id"].isin(kept_users)]


def _wide_matrices(
    mails: pd.DataFrame, mailshot_ids: list[int], n_users: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pivot the long table into (opens, non_opens), each (mailshots x users) float32.

    ``mails["user_id"]`` must already be mapped to column indices. Duplicate
    (mailshot, user) rows are combined with ``max``, i.e. "any".
    """
    indicators = mails[["mailshot_id", "user_id"]].assign(
        opens=(mails["opened"] == 1).astype(np.float32),
        non_opens=(mails["opened"] == 0).astype(np.float32),
    )
    wide = indicators.pivot_table(
        index="mailshot_id",
        columns="user_id",
        values=["opens", "non_opens"],
        aggfunc="max",
        fill_value=0,
    )

    def to_matrix(name: str) -> np.ndarray:
        return (
            wide[name]
            .reindex(index=mailshot_ids, columns=range(n_users), fill_value=0)
            .to_numpy(dtype=np.float32)
        )

    return to_matrix("opens"), to_matrix("non_opens")


def _simulate_time_to_open(mails: pd.DataFrame) -> dict[int, pd.Series]:
    pairs = mails.drop_duplicates(["mailshot_id", "user_id"], keep="last")
    mean_minutes = (
        pd.to_timedelta(pairs["time_to_open"]).dt.total_seconds().abs().to_numpy() / 60
    )

    observed = ~np.isnan(mean_minutes)
    simulated = np.full(len(pairs), np.inf)
    simulated[observed] = np.random.exponential(mean_minutes[observed])

    by_pair = pairs[["mailshot_id", "user_id"]].assign(tto=simulated)
    return {
        int(mailshot_id): group.set_index("user_id")["tto"]
        for mailshot_id, group in by_pair.groupby("mailshot_id")
    }

def _clean_mails(mails: pd.DataFrame) -> pd.DataFrame:
    mails = mails[list(REQUIRED_COLUMNS)]
    mails = mails[
        mails.groupby("mailshot_id")["user_id"].transform("size")
        >= MIN_MAILS_PER_MAILSHOT
    ]
    mails = mails.drop_duplicates(subset=["mailshot_id", "user_id"])
    dense_ids, _ = pd.factorize(mails["mailshot_id"], sort=True)
    return mails.assign(mailshot_id=dense_ids).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class AutoencoderDataset(Dataset):
    def __init__(
        self,
        mails: pd.DataFrame,
        user_id_mapper: dict[int, int] | None = None,
        remove_users_below_n_opens: int = 0,
        sender_id: int | None = None,
    ) -> None:
        missing = set(REQUIRED_COLUMNS) - set(mails.columns)
        if missing:
            raise ValueError(f"`mails` is missing columns: {sorted(missing)}")

        mails = _drop_users_below_min_opens(mails, remove_users_below_n_opens)
        if user_id_mapper is None:
            user_ids = sorted(mails["user_id"].unique().tolist())
            user_id_mapper = {user_id: i for i, user_id in enumerate(user_ids)}
        else:
            mails = mails[mails["user_id"].isin(list(user_id_mapper))]

        self.sender_id = sender_id
        self.user_id_mapper: dict[int, int] = user_id_mapper
        self.mails: pd.DataFrame = mails.assign(
            user_id=mails["user_id"].map(user_id_mapper)
        )
        self.mailshot_ids: list[int] = sorted(self.mails["mailshot_id"].unique().tolist())
        self._row_of_mailshot = {m: row for row, m in enumerate(self.mailshot_ids)}

        self.opens, self.non_opens = _wide_matrices(
            self.mails, self.mailshot_ids, self.num_users
        )
        self._time_to_open = _simulate_time_to_open(self.mails)

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_disk(cls, sender_id: int) -> Self:
        return cls(load_data_for_sender(sender_id), sender_id=sender_id)

    @classmethod
    def from_disk_data_split(
        cls,
        sender_id: int,
        split_sizes: Sequence[int],
        remove_users_below_n_opens: int = 0,
    ) -> list[Self]:

        logger.info("Loading data for sender %s", sender_id)
        mails = _clean_mails(load_data_for_sender(sender_id))

        n_train = mails["mailshot_id"].nunique() - sum(split_sizes)
        if n_train < 1:
            raise ValueError(
                f"split_sizes {list(split_sizes)} leave {n_train} training mailshots"
            )

        edges = [0, *accumulate(split_sizes, initial=n_train)]
        chunks = [
            mails[(mails["mailshot_id"] >= lo) & (mails["mailshot_id"] < hi)]
            for lo, hi in pairwise(edges)
        ]

        train = cls(
            chunks[0],
            remove_users_below_n_opens=remove_users_below_n_opens,
            sender_id=sender_id,
        )
        others = [
            cls(chunk, user_id_mapper=train.user_id_mapper, sender_id=sender_id)
            for chunk in chunks[1:]
        ]
        return [train, *others]

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #
    @property
    def num_users(self) -> int:
        return len(self.user_id_mapper)

    @property
    def shape(self) -> tuple[int, int]:
        return self.opens.shape

    @property
    def min_template_id(self) -> int:
        return self.mailshot_ids[0]

    @property
    def max_template_id(self) -> int:
        return self.mailshot_ids[-1]

    # ------------------------------------------------------------------ #
    # Per-mailshot / per-user accessors
    # ------------------------------------------------------------------ #

    def select_opened_indices(
        self, mailshot_id: int, indices: Sequence[int], time_to_open: float
    ) -> list[int]:

        simulated = self._time_to_open[mailshot_id].reindex(indices)
        return [i for i, tto in zip(indices, simulated) if tto <= time_to_open]

    def mailshot_user_indices(self, mailshot_id: int) -> np.ndarray:
        in_mailshot = self.mails["mailshot_id"] == mailshot_id
        return self.mails.loc[in_mailshot, "user_id"].unique()

    @cached_property
    def sends(self) -> np.ndarray:
        return np.maximum(self.opens, self.non_opens)

    def popularity_tensor(self, template_weight: AbstractTemplateWeight) -> torch.Tensor:
        weights = pd.Series(self.mailshot_ids).map(template_weight).to_numpy(np.float64)
        weighted_opens = weights @ self.opens
        weighted_sends = weights @ self.sends
        rate = np.divide(
            weighted_opens,
            weighted_sends,
            out=np.full(self.num_users, np.nan),
            where=weighted_sends != 0,
        )
        return torch.tensor(rate, dtype=torch.float64)


    def __len__(self) -> int:
        return len(self.opens)

    def __getitem__(self, index):
        x = torch.tensor(self.opens[index])
        return x


class PositiveAutoencoderDataset(Dataset):
    def __init__(self, autoencoder_dataset: AutoencoderDataset, noise: float = 0.0) -> None:
        self.dataset = autoencoder_dataset
        self.noise = noise

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.dataset[index], dtype=torch.float32)
        if self.noise > 0.0:
            mask = torch.bernoulli(torch.full_like(x, 1.0 - self.noise))
            return x * mask, x
        return x, x


class NegativeAutoencoderDataset(Dataset):
    def __init__(self, autoencoder_dataset: AutoencoderDataset, noise: float = 0.0) -> None:
        self.dataset = autoencoder_dataset
        self.noise = noise

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.dataset[index], dtype=torch.float32)
        if self.noise > 0.0:
            mask = torch.bernoulli(torch.full_like(x, 1.0 - self.noise))
            return x * mask, x
        return x, x