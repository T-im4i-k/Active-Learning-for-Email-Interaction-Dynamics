import logging
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from email_outreach.ml.shallow_autoencoder.dataset.autoencoder_dataset import AutoencoderDataset
from email_outreach.ml.shallow_autoencoder.dataset.tto_decay import AbstractTTODecay

logger = logging.getLogger(__name__)


class TTODecayedAutoencoderDataset(Dataset, ABC):
    def __init__(self, autoencoder_dataset: AutoencoderDataset, tto_decay: AbstractTTODecay):
        self.autoencoder_dataset = autoencoder_dataset
        self.tto_decay = tto_decay
        self._y_matrix: np.ndarray = self.tto_decay(
            self.autoencoder_dataset._tto_matrix
        ).astype(np.float32)

    @property
    def y_matrix(self) -> torch.Tensor:
        return torch.tensor(self._y_matrix)

    def __len__(self):
        return len(self.autoencoder_dataset)

    @abstractmethod
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class BinaryToTTOAutoencoderDataset(TTODecayedAutoencoderDataset):
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.autoencoder_dataset._mailshot_embeddings[index])
        y = torch.tensor(self._y_matrix[index])
        return x, y


class TTOToTTOAutoencoderDataset(TTODecayedAutoencoderDataset):
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.tensor(self._y_matrix[index])
        return y, y