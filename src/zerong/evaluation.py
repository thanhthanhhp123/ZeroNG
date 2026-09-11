"""Inference on manifest splits and the metric set reported for every run."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from zerong.metrics.factory import (
    frr_at_recall,
    frr_at_zero_escape,
    rates_at_threshold,
    threshold_from_good_quantile,
)
from zerong.metrics.literature import aupro, image_auroc, pixel_auroc


@dataclass
class SplitPredictions:
    paths: list[str]
    labels: np.ndarray
    scores: np.ndarray
    anomaly_maps: torch.Tensor | None = None  # (N, H, W)
    masks: torch.Tensor | None = None  # (N, H, W)
    forward_seconds: float = 0.0  # model forward only, excluding data loading


@torch.no_grad()
def predict(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device, keep_maps: bool = False
) -> SplitPredictions:
    """Raw scores through the model's full forward (pre-processor + network), as when exported.

    ``forward_seconds`` is a rough training-machine figure (batched, first batch includes
    warm-up). Deployment latency is measured separately by the CPU benchmark.
    """
    model.eval().to(device)
    sync = device.type == "cuda"
    forward_seconds = 0.0
    paths: list[str] = []
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    maps: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for batch in dataloader:
        image = batch.image.to(device)
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(image)
        if sync:
            torch.cuda.synchronize()
        forward_seconds += time.perf_counter() - t0
        n = batch.image.shape[0]
        paths.extend(batch.image_path)
        labels.append(batch.gt_label.cpu().numpy().reshape(-1))
        scores.append(out.pred_score.float().cpu().numpy().reshape(-1))
        if keep_maps and batch.gt_mask is not None:
            gt = batch.gt_mask.reshape(n, *batch.gt_mask.shape[-2:]).cpu()
            amap = out.anomaly_map.float().reshape(n, 1, *out.anomaly_map.shape[-2:])
            if amap.shape[-2:] != gt.shape[-2:]:
                amap = F.interpolate(amap, size=gt.shape[-2:], mode="bilinear", align_corners=False)
            maps.append(amap[:, 0].cpu())
            masks.append(gt)
    return SplitPredictions(
        paths=paths,
        labels=np.concatenate(labels).astype(int),
        scores=np.concatenate(scores),
        anomaly_maps=torch.cat(maps) if maps else None,
        masks=torch.cat(masks) if masks else None,
        forward_seconds=forward_seconds,
    )


def evaluate(preds: dict[str, SplitPredictions], zero_ng_max_frr: float) -> dict[str, float]:
    """Metrics for one run. Thresholds are calibrated on val / val_good and applied to test.

    - ``test_frr_at_zero_escape``: best achievable FRR at 100% NG recall on test (threshold
      chosen on test itself). A property of the score distribution, comparable across models
      like AUROC; not a deployable number.
    - ``valcal_*``: threshold = FRR@zero-escape threshold on val (oracle: uses real defects),
      applied to test. What a client with some real NG images would see.
    - ``zerong_*``: threshold from val_good alone (at most ``zero_ng_max_frr`` false rejects on
      held-out good images), applied to test. The true zero-NG setting.
    """
    test, val, val_good = preds["test"], preds["val"], preds["val_good"]
    m: dict[str, float] = {
        "n_test_good": int(np.sum(test.labels == 0)),
        "n_test_ng": int(np.sum(test.labels == 1)),
        "n_val_good": int(val_good.labels.size),
        "test_image_auroc": image_auroc(test.scores, test.labels),
    }
    if test.anomaly_maps is not None and test.masks is not None:
        m["test_pixel_auroc"] = pixel_auroc(test.anomaly_maps, test.masks)
        m["test_aupro"] = aupro(test.anomaly_maps, test.masks)

    m["test_frr_at_zero_escape"] = frr_at_zero_escape(test.scores, test.labels).frr
    m["test_frr_at_95_recall"] = frr_at_recall(test.scores, test.labels, 0.95).frr

    t_val = frr_at_zero_escape(val.scores, val.labels).threshold
    valcal = rates_at_threshold(test.scores, test.labels, t_val)
    m["valcal_threshold"] = t_val
    m["valcal_test_frr"] = valcal["frr"]
    m["valcal_test_escape_rate"] = valcal["escape_rate"]

    t_zero = threshold_from_good_quantile(val_good.scores, zero_ng_max_frr)
    zerong = rates_at_threshold(test.scores, test.labels, t_zero)
    m["zerong_threshold"] = t_zero
    m["zerong_test_frr"] = zerong["frr"]
    m["zerong_test_escape_rate"] = zerong["escape_rate"]
    return m
