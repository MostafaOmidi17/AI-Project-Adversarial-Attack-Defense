from typing import Sequence

import torch
from torch import Tensor, nn


# مقادیر رایج برای مدل‌های pretrained روی CIFAR-10.
# اگر فایل README مدل شما مقادیر دیگری نوشته، دقیقاً از همان‌ها استفاده کنید.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


class Normalize(nn.Module):
    """
    Normalizes a batch of raw images inside the model.

    Expected input shape:
        (batch_size, 3, height, width)

    Expected input range:
        [0, 1]

    Keeping normalization inside the model allows adversarial attacks
    to operate in the raw pixel space while gradients still pass through
    the normalization operation.
    """

    def __init__(
        self,
        mean: Sequence[float] = CIFAR10_MEAN,
        std: Sequence[float] = CIFAR10_STD,
    ) -> None:
        super().__init__()

        if len(mean) != len(std):
            raise ValueError("mean and std must have the same length.")

        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(
            1, -1, 1, 1
        )
        std_tensor = torch.tensor(std, dtype=torch.float32).view(
            1, -1, 1, 1
        )

        if torch.any(std_tensor <= 0):
            raise ValueError("All standard-deviation values must be positive.")

        # register_buffer باعث می‌شود mean و std همراه مدل به GPU منتقل شوند،
        # اما به‌عنوان پارامتر قابل‌آموزش در نظر گرفته نشوند.
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected a 4D tensor (N, C, H, W), got shape {tuple(x.shape)}."
            )

        if x.shape[1] != self.mean.shape[1]:
            raise ValueError(
                f"Expected {self.mean.shape[1]} channels, got {x.shape[1]}."
            )

        mean = self.mean.to(dtype=x.dtype)
        std = self.std.to(dtype=x.dtype)

        return (x - mean) / std