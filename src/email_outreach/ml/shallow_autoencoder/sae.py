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
    def __init__(self, n: int, d: int, dropout_p=0.0):
        super().__init__()
        self.E = nn.Parameter(torch.empty(n, d))
        self.D = nn.Parameter(torch.empty(n, d))

        init.xavier_uniform_(self.E)
        init.xavier_uniform_(self.D)

        self.dropout = nn.Dropout(p=dropout_p)
        self.bn = nn.LayerNorm(d)
        self.training_metrics = {"train_loss": [], "val_loss": []}

    def logit(self, X):
        diag_vals = (self.E * self.D).sum(dim=1)
        hidden = X @ self.E

        hidden = self.bn(hidden)
        hidden = self.dropout(hidden)

        out = hidden @ self.D.t()
        out = out - (X * diag_vals)

        return out

    def forward(self, X):
        return torch.sigmoid(self.logit(X))

    @staticmethod
    def _weighted_bce(reconstruction, target, positive_weight, criterion):
        """Returns the weighted BCE summed over elements, plus the element count."""
        loss = criterion(reconstruction, target)
        weights = torch.where(
            target > 0,
            torch.full_like(target, positive_weight),
            torch.ones_like(target),
        )
        return (loss * weights).sum(), target.numel()

    @torch.no_grad()
    def _evaluate(self, loader, positive_weight, criterion):
        self.eval()
        total_loss, total_count = 0.0, 0
        for batch_data, target_data in loader:
            reconstruction = self.forward(batch_data)
            loss_sum, count = self._weighted_bce(
                reconstruction, target_data, positive_weight, criterion
            )
            total_loss += loss_sum.item()
            total_count += count
        return total_loss / max(total_count, 1)

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
        set_seed(42)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        criterion = nn.BCELoss(reduction="none")

        self.training_metrics = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss, epoch_count = 0.0, 0

            for batch_data, target_data in train_loader:
                optimizer.zero_grad()

                reconstruction = self.forward(batch_data)
                loss_sum, count = self._weighted_bce(
                    reconstruction, target_data, positive_weight, criterion
                )

                (loss_sum / count).backward()
                optimizer.step()

                epoch_loss += loss_sum.item()
                epoch_count += count

            scheduler.step()

            train_loss = epoch_loss / max(epoch_count, 1)
            val_loss = self._evaluate(val_loader, positive_weight, criterion)

            self.training_metrics["train_loss"].append(train_loss)
            self.training_metrics["val_loss"].append(val_loss)

            logger.info(
                "Epoch %d/%d - train_loss: %.6f - val_loss: %.6f - lr: %.2e",
                epoch,
                epochs,
                train_loss,
                val_loss,
                scheduler.get_last_lr()[0],
            )

        return self.training_metrics

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self.forward(X)

    def predict_logit(self, X):
        self.eval()
        with torch.no_grad():
            return self.logit(X)