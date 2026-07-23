"""2D U-Net for ACDC cardiac segmentation."""

import torch
from monai.networks.nets import UNet

NUM_CLASSES = 4  # background, RV, myocardium, LV


def build_unet(dropout=0.1, channels=(32, 64, 128, 256, 512)):
    """Build the segmentation network.

    A 2D architecture is used deliberately: measured slice thickness is 5 to
    10 mm against an in-plane resolution of 1.5 mm, so 3D convolutions would
    operate over highly anisotropic voxels.

    Dropout is included from the start so that Monte Carlo dropout
    uncertainty estimation in a later phase requires no retraining.
    """
    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=NUM_CLASSES,
        channels=channels,
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=dropout,
    )


def get_device():
    """Pick the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_unet()
    print("parameters:", f"{count_parameters(model):,}")

    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print("forward:", tuple(x.shape), "->", tuple(y.shape))
    assert y.shape == (2, NUM_CLASSES, 256, 256)

    y.mean().backward()
    grads = sum(int(p.grad is not None) for p in model.parameters())
    print("params with gradients:", grads, "of", len(list(model.parameters())))

    model.eval()
    with torch.no_grad():
        a, b = model(x), model(x)
    print("eval deterministic:", torch.allclose(a, b))

    model.train()
    with torch.no_grad():
        a, b = model(x), model(x)
    print("train stochastic (dropout live):", not torch.allclose(a, b))

    print("device selected:", get_device())
