import logging

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from email_outreach.utils.common_utils import set_seed

# References:
# - Sedhain S. et al., "AutoRec: Autoencoders Meet Collaborative Filtering," WWW 2015.
#   https://arxiv.org/abs/1508.01195
# - "Shallow linear autoencoders for collaborative filtering" approaches in related CF literature.

logger = logging.getLogger(__name__)

class ShallowAutoencoder(nn.Module):
    def __init__(
        self,
        n: int,
        d: int,
        dropout_p=0.1,
    ):
        super().__init__()
        self.E = nn.Parameter(torch.empty(n, d))
        self.D = nn.Parameter(torch.empty(n, d))

        init.xavier_uniform_(self.E)
        init.xavier_uniform_(self.D)

        self.dropout = nn.Dropout(p=dropout_p)
        self.bn = nn.LayerNorm(d)

    def logit(self, X):
        diag_vals = (self.E * self.D).sum(dim=1)
        hidden = X @ self.E

        hidden = self.bn(hidden)
        hidden = self.dropout(hidden)

        out = hidden @ self.D.t()
        out = out - (X * diag_vals)

        return out

    def forward(self, X):
        out = self.logit(X)
        return torch.sigmoid(out)

    def fit(
        self,
        train: Dataset,
        val: Dataset,
        epochs,
        lr,
        batch_size,
        weight_decay,
        positive_weight,
    ):
        set_seed()
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        criterion = nn.BCELoss(reduction="none")

        for _ in range(epochs):
            self.train()
            for batch_data, target_data in train_loader:
                optimizer.zero_grad()

                reconstruction = self.forward(batch_data)
                loss = criterion(reconstruction, target_data)

                weights = torch.where(target_data > 0, positive_weight, 1)
                loss.backward(weights)
                optimizer.step()
            scheduler.step()

        return self.training_metrics

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self.forward(X)
        return reconstruction

    def predict_logit(self, X):
        self.eval()
        with torch.no_grad():
            return self.logit(X)
