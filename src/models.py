from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn

try:
    # حالت استاندارد: from src.models import ...
    from .normalization import (
        CIFAR10_MEAN,
        CIFAR10_STD,
        Normalize,
    )
except ImportError:
    # در صورتی که خود پوشه src مستقیماً به sys.path اضافه شده باشد.
    from normalization import (
        CIFAR10_MEAN,
        CIFAR10_STD,
        Normalize,
    )


def _conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    """
    Basic residual block used by CIFAR-10 ResNet-20.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = _conv3x3(
            in_channels,
            out_channels,
            stride=stride,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = _conv3x3(
            out_channels,
            out_channels,
            stride=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                _conv1x1(
                    in_channels,
                    out_channels,
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet20(nn.Module):
    """
    Standard ResNet-20 architecture for 32x32 CIFAR-10 images.

    Architecture:
        Initial convolution
        3 residual blocks with 16 channels
        3 residual blocks with 32 channels
        3 residual blocks with 64 channels
        Global average pooling
        Fully-connected output layer
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.in_channels = 16

        self.conv1 = _conv3x3(
            in_channels=3,
            out_channels=16,
            stride=1,
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(
            out_channels=16,
            num_blocks=3,
            stride=1,
        )
        self.layer2 = self._make_layer(
            out_channels=32,
            num_blocks=3,
            stride=2,
        )
        self.layer3 = self._make_layer(
            out_channels=64,
            num_blocks=3,
            stride=2,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        self._initialize_weights()

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        blocks = [
            BasicBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
            )
        ]

        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            blocks.append(
                BasicBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1,
                )
            )

        return nn.Sequential(*blocks)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        logits = self.fc(out)

        return logits


def resnet20(num_classes: int = 10) -> ResNet20:
    return ResNet20(num_classes=num_classes)


def _extract_state_dict(checkpoint: Any) -> Mapping[str, Tensor]:
    """
    Extracts a state_dict from common checkpoint formats.
    """

    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Checkpoint must be a state_dict, dictionary, or nn.Module."
        )

    common_keys = (
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "network",
    )

    for key in common_keys:
        if key not in checkpoint:
            continue

        candidate = checkpoint[key]

        if isinstance(candidate, nn.Module):
            return candidate.state_dict()

        if isinstance(candidate, Mapping):
            return candidate

    # اگر خود checkpoint همان state_dict باشد.
    if all(isinstance(key, str) for key in checkpoint.keys()):
        return checkpoint

    raise ValueError("Could not find a valid state_dict in checkpoint.")


def _clean_state_dict(
    state_dict: Mapping[str, Tensor],
) -> OrderedDict[str, Tensor]:
    """
    Cleans common prefixes and layer-name differences.

    Supported examples:
        module.conv1.weight
        model.conv1.weight
        backbone.conv1.weight
        linear.weight -> fc.weight
        shortcut.0.weight -> downsample.0.weight
    """

    cleaned = OrderedDict(
        (str(key), value)
        for key, value in state_dict.items()
        if torch.is_tensor(value)
    )

    # حالتی که وزن کل مدل Sequential ذخیره شده باشد:
    # normalize.* و backbone.*
    if any(key.startswith("backbone.") for key in cleaned):
        cleaned = OrderedDict(
            (
                key.removeprefix("backbone."),
                value,
            )
            for key, value in cleaned.items()
            if key.startswith("backbone.")
        )

    # حالتی که مدل با nn.Sequential ذخیره شده باشد:
    # 0 = Normalize و 1 = ResNet20
    elif any(key.startswith("1.conv1.") for key in cleaned):
        cleaned = OrderedDict(
            (
                key.removeprefix("1."),
                value,
            )
            for key, value in cleaned.items()
            if key.startswith("1.")
        )

    # حذف پیشوندهای رایج.
    for prefix in ("module.", "model.", "net.", "network."):
        if cleaned and all(
            key.startswith(prefix) for key in cleaned.keys()
        ):
            cleaned = OrderedDict(
                (
                    key.removeprefix(prefix),
                    value,
                )
                for key, value in cleaned.items()
            )

    remapped = OrderedDict()

    for key, value in cleaned.items():
        # برخی پیاده‌سازی‌ها classifier را linear نام‌گذاری می‌کنند.
        if key.startswith("linear."):
            key = "fc." + key.removeprefix("linear.")

        if key.startswith("classifier."):
            key = "fc." + key.removeprefix("classifier.")

        # برخی پیاده‌سازی‌ها اتصال residual را shortcut می‌نامند.
        key = key.replace(".shortcut.", ".downsample.")

        remapped[key] = value

    return remapped


def load_pretrained_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # برای نسخه‌های قدیمی‌تر PyTorch که weights_only ندارند.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    state_dict = _extract_state_dict(checkpoint)
    state_dict = _clean_state_dict(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint architecture does not exactly match this ResNet-20 "
            "implementation. Do not use strict=False, because that may leave "
            "parts of the model randomly initialized.\n\n"
            f"Original loading error:\n{error}"
        ) from error

    return model


def build_normalized_resnet20(
    checkpoint_path: Optional[str | Path] = None,
    device: Optional[torch.device | str] = None,
    num_classes: int = 10,
    mean: tuple[float, float, float] = CIFAR10_MEAN,
    std: tuple[float, float, float] = CIFAR10_STD,
    eval_mode: bool = True,
) -> nn.Module:
    """
    Builds the complete model:

        Raw image [0,1] -> Normalize -> ResNet20 -> logits
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    backbone = resnet20(num_classes=num_classes)

    if checkpoint_path is not None:
        backbone = load_pretrained_weights(
            backbone,
            checkpoint_path,
        )

    model = nn.Sequential(
        OrderedDict(
            [
                ("normalize", Normalize(mean=mean, std=std)),
                ("backbone", backbone),
            ]
        )
    )

    model = model.to(device)

    if eval_mode:
        model.eval()

    return model