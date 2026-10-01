import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet import UNet


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=32,
    ).to(device)

    model.eval()

    x = torch.randn(
        2,
        3,
        256,
        256,
        device=device,
    )

    with torch.no_grad():
        y = model(x)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("U-Net forward test successful")
    print("=" * 50)
    print("Device:", device)
    print("GPU:", (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    ))
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Output dtype:", y.dtype)
    print("Parameters:", f"{parameters:,}")
    print("=" * 50)


if __name__ == "__main__":
    main()