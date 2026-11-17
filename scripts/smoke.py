"""One seeded optimizer step; proves wiring, not prediction quality."""

import json
import torch
from tensornet.model import DSResNet
from tensornet.model.loss import rmse_loss, bce_dice_loss

torch.manual_seed(42)
torch.set_num_threads(2)
model = DSResNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
depth, mask = model({"bg": torch.rand(1, 3, 64, 64), "bg_fg": torch.rand(1, 3, 64, 64)})
loss = rmse_loss()(depth, torch.rand_like(depth)) + bce_dice_loss()(
    mask, torch.zeros_like(mask)
)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(
    json.dumps(
        {
            "fixture": "synthetic; no accuracy claim",
            "depth_shape": list(depth.shape),
            "mask_shape": list(mask.shape),
            "loss": loss.item(),
            "optimizer_step": "passed",
        }
    )
)
