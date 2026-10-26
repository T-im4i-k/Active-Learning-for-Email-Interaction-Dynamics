from collections import defaultdict
from typing import Sequence

import numpy as np

from email_outreach.sae_based_mab.config import ContextualBanditMetrics


def _group_by_mailshot(metrics: list[ContextualBanditMetrics]):
    by_mailshot = defaultdict(list)
    for m in metrics:
        if m.present_in_prediction:
            by_mailshot[m.mailshot_id].append(m)
    return by_mailshot


def calculate_recall_precision_at_quantiles(
    metrics: list[ContextualBanditMetrics],
    quantiles: Sequence[float] = (0.05, 0.15, 0.25, 0.35, 0.5, 0.75, 0.85),
) -> tuple[list[float], list[float], list[float]]:
    q = np.asarray(quantiles, dtype=np.float64)
    rec_sum = np.zeros(q.size, dtype=np.float64)
    prec_sum = np.zeros(q.size, dtype=np.float64)
    weight_total = 0

    for ms in _group_by_mailshot(metrics).values():
        n = len(ms)
        y_true = np.fromiter((m.opened for m in ms), dtype=np.int8, count=n)
        y_pred = np.fromiter(
            (m.prediction - m.stage for m in ms), dtype=np.float64, count=n
        )
        pos = int(y_true.sum())
        if pos == 0:
            continue

        order = np.argsort(y_pred)[::-1]
        cum_tp = np.cumsum(y_true[order], dtype=np.int64)

        k = np.ceil(np.round((1.0 - q) * n, 9)).astype(int) - 1
        k = np.clip(k, 0, n - 1)

        tp_k = cum_tp[k]
        rec_sum += (tp_k / pos) * n
        prec_sum += (tp_k / (k + 1)) * n
        weight_total += n

    sent_pct = (1.0 - q).tolist()
    if weight_total == 0:
        zeros = [0.0] * q.size
        return sent_pct, zeros, zeros

    return (
        sent_pct,
        (rec_sum / weight_total).tolist(),
        (prec_sum / weight_total).tolist(),
    )


def calculate_auc(metrics: list[ContextualBanditMetrics]) -> float:
    area_sum = 0.0
    weight_total = 0

    for ms in _group_by_mailshot(metrics).values():
        n = len(ms)
        y_true = np.fromiter((m.opened for m in ms), dtype=np.int8, count=n)
        pos = int(y_true.sum())
        if pos == 0:
            continue

        y_pred = np.fromiter(
            (m.prediction - m.stage for m in ms), dtype=np.float64, count=n
        )
        order = np.argsort(y_pred)[::-1]
        recall = np.cumsum(y_true[order], dtype=np.int64) / pos

        area = (recall.sum() - recall[-1] / 2) / n
        area_sum += area * n
        weight_total += n

    return area_sum / weight_total if weight_total else 0.0
