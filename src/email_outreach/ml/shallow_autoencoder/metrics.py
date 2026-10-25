from collections import defaultdict
from typing import Sequence

import numpy as np

from email_outreach.ml.shallow_autoencoder.config import ContextualBanditMetrics


def calculate_recall_precision_at_quantiles(
    metrics: list[ContextualBanditMetrics],
    quantiles: Sequence[float] = (0.25, 0.5, 0.75, 0.85),
) -> tuple[list[float], list[float], list[float]]:

    by_mailshot = defaultdict(list)
    for m in metrics:
        if not m.present_in_prediction:
            continue
        by_mailshot[m.mailshot_id].append(m)

    q = np.asarray(quantiles, dtype=np.float64)
    n_q = q.size
    rec_sum = np.zeros(n_q, dtype=np.float64)
    prec_sum = np.zeros(n_q, dtype=np.float64)
    weights = np.zeros(n_q, dtype=np.float64)

    for ms in by_mailshot.values():
        y_true = np.fromiter((m.opened for m in ms), dtype=np.int8)
        y_pred = np.fromiter(
            (m.prediction - m.stage for m in ms),
            dtype=np.float64,
        )
        pos = y_true.sum()
        if pos == 0:
            continue

        order = np.argsort(y_pred)[::-1]
        y_true_sorted = y_true[order]
        cum_tp = np.cumsum(y_true_sorted, dtype=np.int64)

        k = np.ceil((1.0 - q) * len(y_pred)).astype(int) - 1
        k = np.clip(k, 0, len(y_pred) - 1)

        tp_k = cum_tp[k]
        rec_k = tp_k / pos
        pre_k = tp_k / (k + 1)

        w = len(ms)
        rec_sum += rec_k * w
        prec_sum += pre_k * w
        weights += w

    nz = weights > 0
    final_rec = np.zeros_like(rec_sum)
    final_prec = np.zeros_like(prec_sum)
    final_rec[nz] = rec_sum[nz] / weights[nz]
    final_prec[nz] = prec_sum[nz] / weights[nz]

    sent_pct = (1.0 - q).tolist()
    return sent_pct, final_rec.tolist(), final_prec.tolist()


def calculate_auc(metrics: list[ContextualBanditMetrics]) -> float:
    by_mailshot = defaultdict(list)
    for m in metrics:
        if not m.present_in_prediction:
            continue
        by_mailshot[m.mailshot_id].append(m)

    area_sum = 0.0
    weight_total = 0

    for ms in by_mailshot.values():
        n = len(ms)
        y_true = np.fromiter((m.opened for m in ms), dtype=np.int8, count=n)
        pos = int(y_true.sum())
        if pos == 0:
            continue

        y_pred = np.fromiter(
            (m.prediction - m.stage for m in ms),
            dtype=np.float64,
            count=n,
        )

        order = np.argsort(y_pred)[::-1]
        cum_tp = np.cumsum(y_true[order], dtype=np.int64)

        area_sum += cum_tp.sum() / pos
        weight_total += n

    return area_sum / weight_total if weight_total else 0.0
