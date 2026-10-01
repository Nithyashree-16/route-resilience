import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.losses import (
    DiceLoss,
    SoftIoULoss,
    BoundaryLoss,
    BaselineRoadLoss,
    AdvancedRoadLoss,
)


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logits = torch.randn(
        2,
        1,
        256,
        256,
        device=device,
        requires_grad=True,
    )

    targets = torch.randint(
        0,
        2,
        (
            2,
            1,
            256,
            256,
        ),
        device=device,
        dtype=torch.float32,
    )

    losses = {
        "Dice": DiceLoss().to(device),
        "Soft IoU": SoftIoULoss().to(device),
        "Boundary": BoundaryLoss().to(device),
        "Baseline": BaselineRoadLoss().to(device),
        "Advanced": AdvancedRoadLoss().to(device),
    }

    print("Loss test")
    print("=" * 50)

    for name, loss_fn in losses.items():
        value = loss_fn(
            logits,
            targets,
        )

        print(
            f"{name:10} = {value.item():.6f}"
        )

    combined = BaselineRoadLoss().to(device)
    value = combined(
        logits,
        targets,
    )

    value.backward()

    print("=" * 50)
    print("Backward pass successful")
    print("Device:", device)


if __name__ == "__main__":
    main()