import torch
from torch.utils.data import Dataset
from email_outreach.ml.shallow_autoencoder.dataset.autoencoder_dataset import AutoencoderDataset
import numpy as np

class NoisyDataset(Dataset):
    def __init__(self, autoencoder_dataset: AutoencoderDataset, p_min: float, p_max: float):
        self.autoencoder_dataset = autoencoder_dataset

        if not 0 <= p_min <= p_max <= 1:
            raise ValueError("Invalid p")

        self.p_min = p_min
        self.p_max = p_max


    def __len__(self):
        return len(self.autoencoder_dataset._mailshot_embeddings)

    def __getitem__(self, index):
        p = np.random.uniform(self.p_min, self.p_max)

        x = torch.tensor(self.autoencoder_dataset._mailshot_embeddings[index])
        mask = torch.empty_like(x).bernoulli(p=p)
        x_noisy = mask * x


        return x_noisy, x