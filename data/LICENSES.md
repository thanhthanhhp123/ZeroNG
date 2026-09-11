# Dataset licenses

Every dataset used by ZeroNG is recorded here **before first use**, with the source checked on the
date given. Datasets themselves live in `data/` and are never committed.

| Dataset | License | Commercial use | Source (checked 2026-09-11) |
|---|---|---|---|
| MVTec AD | CC BY-NC-SA 4.0 | **No** | https://www.mvtec.com/company/research/datasets/mvtec-ad |
| MVTec AD 2 | CC BY-NC-SA 4.0 | **No** | https://www.mvtec.com/company/research/datasets/mvtec-ad-2 |
| VisA | CC BY 4.0 (data); Apache-2.0 (code) | Yes, with attribution | https://github.com/amazon-science/spot-diff |
| Imagenette (auxiliary) | Repo Apache-2.0; images are an ImageNet subset, subject to ImageNet terms of access | **No** (ImageNet terms: non-commercial research) | https://github.com/fastai/imagenette |

Imagenette is not an evaluation dataset: it only feeds EfficientAD's ImageNet penalty term during
training (as in the original paper).

Note: anomalib 2.6.1's VisA docstring says CC BY-NC-SA 4.0; the dataset authors' repository says
CC BY 4.0. The authors' statement is recorded above.

Not yet verified (do not download until a row is added above): MVTec LOCO AD, Real-IAD,
KolektorSDD2, Severstal, NEU-DET.

MVTec AD 2: the MVTec page asks users to fill in a form before downloading. By the project owner's
decision (2026-09-11), ZeroNG downloads the archive from the URL and SHA-256 pinned in anomalib
2.6.1 instead; the CC BY-NC-SA 4.0 terms apply unchanged. Only `test_public` has public ground
truth; `test_private` and `test_private_mixed` are scored by https://benchmark.mvtec.com/.

## Consequences for this project

- Because MVTec AD is non-commercial, models trained on it and any results derived from it are for
  research and portfolio purposes only. The README states this.
- ShareAlike: redistributed derivatives of MVTec AD (e.g. corrupted test images) must keep
  CC BY-NC-SA 4.0. ZeroNG regenerates corruptions from a seed instead of redistributing images.

## Citations

- Bergmann et al., "MVTec AD – A Comprehensive Real-World Dataset for Unsupervised Anomaly
  Detection", CVPR 2019; extended version in IJCV 2021.
- Zou et al., "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and
  Segmentation", ECCV 2022 (arXiv:2207.14315).
