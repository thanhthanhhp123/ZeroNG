"""Literature metrics (image/pixel AUROC, AUPRO) for comparability with published results."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score


def image_auroc(scores: ArrayLike, labels: ArrayLike) -> float:
    return float(roc_auc_score(np.asarray(labels), np.asarray(scores, dtype=np.float64)))


def pixel_auroc(anomaly_maps: torch.Tensor, masks: torch.Tensor) -> float:
    """AUROC over all pixels of all images. ``anomaly_maps`` and ``masks`` are (N, H, W)."""
    if anomaly_maps.shape != masks.shape:
        raise ValueError(f"shape mismatch: {tuple(anomaly_maps.shape)} vs {tuple(masks.shape)}")
    target = (masks > 0).reshape(-1).numpy()
    preds = anomaly_maps.reshape(-1).float().numpy()
    return float(roc_auc_score(target, preds))


def aupro(anomaly_maps: torch.Tensor, masks: torch.Tensor, fpr_limit: float = 0.3) -> float:
    """Area under the per-region-overlap curve up to ``fpr_limit``, normalised to [0, 1].

    Delegates to anomalib's implementation (pinned version) so numbers match the benchmark
    tables produced with anomalib.
    """
    from anomalib.metrics.aupro import _AUPRO

    if anomaly_maps.shape != masks.shape:
        raise ValueError(f"shape mismatch: {tuple(anomaly_maps.shape)} vs {tuple(masks.shape)}")
    metric = _AUPRO(fpr_limit=fpr_limit)
    metric.update(anomaly_maps.float(), (masks > 0).int())
    return float(metric.compute())
