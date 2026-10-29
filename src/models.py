"""Model factory: simple_cnn (from scratch), resnet18 and mobilenet_v3_small (scratch or ImageNet-pretrained).

All models take a normalized [B, 3, 64, 64] batch. The torchvision models are adapted
to the small input so that the last feature map is not just 2x2:
- resnet18: conv1 stride 2 -> 1 and the max pool removed (64 -> 8x8 at layer4),
- mobilenet_v3_small: stride of the first convolution 2 -> 1 (64 -> 4x4 at the last conv).
Only strides change, so ImageNet weights load unchanged.

Run as a module to print parameter counts and feature map sizes:
    python -m src.models
"""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torchvision import models as tvm

from src.data import IMG_SIZE, NUM_CLASSES

MODEL_NAMES = ("simple_cnn", "resnet18", "mobilenet_v3_small")


class SimpleCNN(nn.Module):
    """Baseline: 3 blocks [Conv3x3-BN-ReLU, Conv3x3-BN-ReLU, MaxPool2] -> GAP -> Dropout -> Linear."""

    def __init__(self, num_classes: int = NUM_CLASSES, channels: tuple[int, ...] = (32, 64, 128),
                 dropout: float = 0.3):
        super().__init__()
        blocks, in_ch = [], 3
        for ch in channels:
            blocks.append(nn.Sequential(_conv_bn_relu(in_ch, ch), _conv_bn_relu(ch, ch), nn.MaxPool2d(2)))
            in_ch = ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch),
                         nn.ReLU(inplace=True))


def build_model(name: str, pretrained: bool = False, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Build a model with a fresh `num_classes` head. pretrained=True loads IMAGENET1K_V1 weights."""
    if name == "simple_cnn":
        if pretrained:
            raise ValueError("simple_cnn is trained from scratch only")
        return SimpleCNN(num_classes)
    if name == "resnet18":
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.conv1.stride = (1, 1)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "mobilenet_v3_small":
        model = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        model.features[0][0].stride = (1, 1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"model must be one of {MODEL_NAMES}, got {name!r}")


def feature_extractor(model: nn.Module) -> nn.Module:
    """The part of the model up to (excluding) global pooling; its output is the last feature map."""
    if isinstance(model, SimpleCNN):
        return model.features
    if isinstance(model, tvm.ResNet):
        return nn.Sequential(*list(model.children())[:-2])
    if isinstance(model, tvm.MobileNetV3):
        return model.features
    raise TypeError(f"unsupported model type {type(model).__name__}")


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Last convolutional block, the usual Grad-CAM target (pytorch-grad-cam `target_layers=[...]`)."""
    if isinstance(model, SimpleCNN):
        return model.features[-1][1]  # last Conv-BN-ReLU, before the final max pool (16x16)
    if isinstance(model, tvm.ResNet):
        return model.layer4[-1]
    if isinstance(model, tvm.MobileNetV3):
        return model.features[-1]  # 1x1 conv 96 -> 576 + BN + Hardswish
    raise TypeError(f"unsupported model type {type(model).__name__}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def output_shape(module: nn.Module, model: nn.Module, size: int = IMG_SIZE) -> tuple[int, ...]:
    """Shape (C, H, W) of `module`'s output when `model` runs on one size x size image."""
    shapes = []
    handle = module.register_forward_hook(lambda _m, _i, out: shapes.append(tuple(out.shape[1:])))
    model.eval()(torch.zeros(1, 3, size, size))
    handle.remove()
    return shapes[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Print parameter counts and feature map sizes of all models.")
    parser.add_argument("--no-pretrained", action="store_true", help="skip loading (downloading) ImageNet weights")
    args = parser.parse_args()

    variants = [("simple_cnn", False), ("resnet18", False), ("mobilenet_v3_small", False)]
    if not args.no_pretrained:
        variants += [("resnet18", True), ("mobilenet_v3_small", True)]
    print(f"input: 3x{IMG_SIZE}x{IMG_SIZE}, classes: {NUM_CLASSES}")
    print(f"{'model':<20} {'init':<10} {'params':>11} {'last feature map':>18} {'Grad-CAM layer output':>22}")
    for name, pretrained in variants:
        model = build_model(name, pretrained)
        x = torch.zeros(2, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            logits = model.eval()(x)
            fmap = tuple(feature_extractor(model)(x).shape[1:])
        assert logits.shape == (2, NUM_CLASSES), logits.shape
        cam = output_shape(gradcam_target_layer(model), model)
        print(f"{name:<20} {'ImageNet' if pretrained else 'scratch':<10} {count_params(model):>11,} "
              f"{'x'.join(map(str, fmap)):>18} {'x'.join(map(str, cam)):>22}")


if __name__ == "__main__":
    main()
