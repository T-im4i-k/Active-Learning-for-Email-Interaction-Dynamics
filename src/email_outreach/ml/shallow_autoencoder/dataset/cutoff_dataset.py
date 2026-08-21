import torch
from torch.utils.data import Dataset
from email_outreach.ml.shallow_autoencoder.dataset.autoencoder_dataset import AutoencoderDataset

class CutoffDataset(Dataset):
    def __init__(self, autoencoder_dataset: AutoencoderDataset, cutoff_minutes: float):
        self.autoencoder_dataset = autoencoder_dataset
        self.cutoff_minutes = cutoff_minutes

        self._cutoff_dataset = torch.tensor(self.autoencoder_dataset._tto_matrix <= cutoff_minutes, dtype=torch.float32)

    def __len__(self):
        return len(self._cutoff_dataset)

    def __getitem__(self, index):
        x = torch.tensor(self._cutoff_dataset[index])
        return x, x