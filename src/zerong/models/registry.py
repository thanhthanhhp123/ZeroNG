"""Build anomalib models from ZeroNG configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anomalib.models import EfficientAd, Patchcore
from anomalib.models.components import AnomalibModule

MODELS = ("patchcore", "efficientad")


def build_model(
    name: str, params: dict[str, Any], image_size: int, aux_data_root: Path
) -> AnomalibModule:
    """Instantiate a model that outputs raw anomaly scores.

    anomalib's post-processing (score normalisation, adaptive threshold), evaluator and visualizer
    are disabled: thresholds and metrics are computed by ZeroNG on raw scores, under the split
    protocol, so no test data influences them.
    """
    params = dict(params)
    components = {"post_processor": False, "evaluator": False, "visualizer": False}
    if name == "patchcore":
        cls = Patchcore
    elif name == "efficientad":
        cls = EfficientAd
        params.setdefault("imagenet_dir", str(Path(aux_data_root) / "imagenette"))
    else:
        raise ValueError(f"unknown model {name!r}; expected one of {MODELS}")
    model = cls(**params, **components)
    # Set after construction rather than passed to __init__: anomalib's save_hyperparameters()
    # would store the PreProcessor object in every checkpoint, and pickling it reaches the model
    # through its bound methods. EfficientAd.__getstate__ does not drop the trainer (Lightning's
    # own __getstate__ does), so the checkpoint pickle then pulls in the trainer and its live
    # DataLoader worker iterators and fails.
    model.pre_processor = cls.configure_pre_processor((image_size, image_size))
    return model
