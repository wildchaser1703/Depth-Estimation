# Paired Depth Estimation and Segmentation

A TensorNet-based research study with paired background/composite RGB inputs and separate depth and segmentation decoders. This is **not** a general-purpose single-image depth estimator.

## Attribution

The included library derives from [TensorNet v0.9](https://github.com/shan18/TensorNet), as recorded in `docs/project_setup.md`. Preserve its source notices and the existing licence. SSIM and model-summary helpers also retain their upstream credits. The original notebooks document the study; the framework is not presented as entirely original work.

## Reproducible CPU setup (Python 3.11 or 3.12)

```sh
python -m venv .venv
. .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt
python -m pytest -q
python scripts/smoke.py
```

The PyTorch CPU wheel host is `download.pytorch.org`; it avoids downloading CUDA dependencies. TLS verification stays enabled. The smoke command uses random paired images and performs one optimizer step. Tests exercise both heads, backward gradients, checkpoint reload, input validation, a complete tiny training/validation run, and rejection of overlapping scenes. These checks prove runnable model wiring, **not depth accuracy**.

## Train on your data

```sh
python -m tensornet.training --train train.npz --validation validation.npz --epochs 10
```

Each NPZ must contain numeric arrays `bg`, `bg_fg` of shape N×3×H×W; `depth`, `mask` of shape N×1×H×W; and a one-dimensional `scene_ids` array. Scale RGB, normalized depth and masks to [0,1]. H/W must be multiples of 32 and at least 64. Split by original scene/background before generating composites; scene IDs must not overlap between training and validation. The loader rejects missing, nonfinite or misaligned arrays. Files must use non-object NumPy arrays (`allow_pickle=False`).

The minimal trainer saves a model checkpoint and held-out normalized-depth RMSE/mask IoU with seed and input hashes under `reports/latest`. It is independent of the legacy TensorNet learner and notebook dependencies. The included historical notebooks require their original datasets and optional visualization/augmentation libraries; they are not part of the CPU quickstart. Full historical training and reported benchmark scores have not been reproduced. No pretrained checkpoint or dataset is bundled.

## CI and tested dependencies

`requirements.lock.txt` records the isolated Linux/Python 3.12 test environment. CI installs that lock, runs `pip check`, and executes the offline suite on pushes and pull requests; it does not deploy. For exact replay, use the lock in place of `requirements.txt` (Depth still needs the CPU Torch install first), then `pip install --no-deps -e .` for packaged repos. No credentials are required by CI.
