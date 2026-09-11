import pytest
import torch

from zerong.metrics.literature import aupro, image_auroc, pixel_auroc


def test_image_auroc():
    assert image_auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert image_auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert image_auroc([0.1, 0.5, 0.4, 0.9], [0, 0, 1, 1]) == 0.75


def _masks():
    masks = torch.zeros(4, 32, 32, dtype=torch.uint8)
    masks[1, 4:12, 4:12] = 1
    masks[2, 20:30, 10:20] = 1
    masks[2, 2:6, 2:6] = 1  # second region in the same image
    return masks


def test_pixel_metrics_perfect_map():
    masks = _masks()
    maps = masks.float() + 0.01 * torch.rand(
        masks.shape, generator=torch.Generator().manual_seed(0)
    )
    assert pixel_auroc(maps, masks) == 1.0
    assert aupro(maps, masks) == pytest.approx(1.0, abs=1e-3)


def test_pixel_metrics_random_map_is_poor():
    masks = _masks()
    maps = torch.rand(masks.shape, generator=torch.Generator().manual_seed(0))
    assert 0.35 < pixel_auroc(maps, masks) < 0.65
    assert aupro(maps, masks) < 0.3


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        pixel_auroc(torch.rand(2, 8, 8), torch.zeros(2, 8, 9))
    with pytest.raises(ValueError):
        aupro(torch.rand(2, 8, 8), torch.zeros(2, 8, 9))
