import torch
import pytest
from tensornet.model import DSResNet
from tensornet.model.loss import bce_dice_loss, rmse_loss

torch.set_num_threads(2)


def test_two_heads_backward_and_checkpoint(tmp_path):
    torch.manual_seed(42)
    model = DSResNet()
    batch = {"bg": torch.rand(1, 3, 64, 64), "bg_fg": torch.rand(1, 3, 64, 64)}
    depth, mask = model(batch)
    assert depth.shape == mask.shape == (1, 1, 64, 64)
    loss = rmse_loss()(depth, torch.rand_like(depth)) + bce_dice_loss()(
        mask, torch.zeros_like(mask)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.D0.weight.grad.abs().sum() > 0
    assert model.M0.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        expected = model(batch)
    path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    restored = DSResNet().eval()
    restored.load(path)
    with torch.no_grad():
        actual = restored(batch)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize(
    "batch",
    [
        {},
        {"bg": torch.rand(1, 3, 63, 64), "bg_fg": torch.rand(1, 3, 63, 64)},
        {"bg": torch.rand(1, 3, 64, 64), "bg_fg": torch.rand(1, 1, 64, 64)},
    ],
)
def test_invalid_inputs(batch):
    with pytest.raises(ValueError):
        DSResNet()(batch)


def test_rmse_smoothing_is_respected():
    x = torch.zeros(1)
    torch.testing.assert_close(rmse_loss(smooth=0.25)(x, x), torch.tensor(0.5))
