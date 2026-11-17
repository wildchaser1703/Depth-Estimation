import numpy as np
import pytest
from tensornet.training import train


def write(path, scene):
    rng = np.random.default_rng(42)
    np.savez(
        path,
        bg=rng.random((1, 3, 64, 64)),
        bg_fg=rng.random((1, 3, 64, 64)),
        depth=rng.random((1, 1, 64, 64)),
        mask=np.zeros((1, 1, 64, 64)),
        scene_ids=np.array([scene]),
    )


def test_training_and_validation(tmp_path):
    a, b = tmp_path / "train.npz", tmp_path / "val.npz"
    write(a, "scene-a")
    write(b, "scene-b")
    result = train(a, b, tmp_path / "output")
    assert np.isfinite(result["normalized_depth_rmse"])
    assert 0 <= result["mask_iou"] <= 1
    assert (tmp_path / "output/model.pt").is_file()


def test_shared_backgrounds_rejected(tmp_path):
    a, b = tmp_path / "train.npz", tmp_path / "val.npz"
    write(a, "same")
    write(b, "same")
    with pytest.raises(ValueError, match="share scenes"):
        train(a, b, tmp_path / "output")
