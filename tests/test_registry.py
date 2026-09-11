import pickle
from pathlib import Path

import pytest
import torch
from anomalib.data.transforms.utils import extract_transforms_by_type
from torchvision.transforms.v2 import Resize

from zerong.models.registry import build_model


@pytest.mark.parametrize(
    ("name", "params"), [("patchcore", {"pre_trained": False}), ("efficientad", {})]
)
def test_pre_processor_size_and_picklable_hparams(name: str, params: dict, tmp_path: Path):
    model = build_model(name, params, image_size=128, aux_data_root=tmp_path)
    resize = extract_transforms_by_type(model.pre_processor.transform, Resize)[0]
    assert tuple(resize.size) == (128, 128)
    # Regression: a PreProcessor object in the hyper-parameters makes EfficientAD checkpoints
    # pickle the trainer and fail on live DataLoader worker iterators.
    assert not isinstance(model.hparams.get("pre_processor"), torch.nn.Module)
    pickle.dumps(dict(model.hparams))


def test_unknown_model(tmp_path: Path):
    with pytest.raises(ValueError):
        build_model("yolo", {}, 256, tmp_path)
