import logging
from dataclasses import dataclass

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


@dataclass
class TrainingMetrics:
    loss: float
    average_train_ndcg: float | None
    average_ndcg: float | None
    average_f1: float | None
    recall: float | None
    precision: float | None
    val_loss: float | None
    val_average_ndcg: float | None
    val_average_f1: float | None
    val_average_iou: float | None

    def __str__(self):
        return f"loss: {self.loss:.4f}, val (masked) NDCG: {self.average_ndcg:.4f}, val (masked) f1: {self.average_f1:.4f}, val loss: {self.val_loss:.4f}, val NDCG: {self.val_average_ndcg:.4f}, val f1: {self.val_average_f1:.4f}"

class ShallowAutoencoder(nn.Module):
    """
    Shallow linear autoencoder based on the formulation:

       B_{E,D} = E D^T - diag([E ⊙ D] 1_n)
       X B_{E,D} = X E D^T - X diag([E ⊙ D] 1_n)

    E, D in R^{n x d} are trainable parameters (n = #items or #features, d = latent dimension).
    diag(...) ensures the diagonal is zeroed out, avoiding trivial solutions.

    n - usually around 100K (users or features)
    d - latent dimension, usually 8-20
    number of rows usually 100 templates
    """

    def __init__(
            self, n: int, d: int,  dropout_p=0.1,
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
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        criterion = nn.BCELoss(reduction="none")

        for epoch in range(epochs):
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