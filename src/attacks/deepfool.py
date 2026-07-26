from typing import Final, Optional

import math
import torch
from torch import Tensor, nn


__all__ = ["deepfool_attack"]


# مقدار کوچکی که باعث می‌شود نمونه کمی از مرز تصمیم عبور کند.
_BOUNDARY_OFFSET: Final[float] = 1e-4

# جلوگیری از تقسیم بر صفر در حالتی که گرادیان دو کلاس تقریباً یکسان است.
_MIN_GRADIENT_NORM: Final[float] = 1e-12


def _validate_inputs(
    images: Tensor,
    labels: Tensor,
    max_steps: int,
    overshoot: float,
    num_classes: int,
) -> None:
    """
    Validates the common DeepFool inputs.
    """

    if images.ndim != 4:
        raise ValueError(
            "Expected images with shape (N, C, H, W), "
            f"got {tuple(images.shape)}."
        )

    if images.shape[0] == 0:
        raise ValueError(
            "images must contain at least one sample."
        )

    if not torch.is_floating_point(images):
        raise TypeError(
            "images must be a floating-point tensor."
        )

    expected_labels_shape = (images.shape[0],)

    if (
        labels.ndim != 1
        or tuple(labels.shape) != expected_labels_shape
    ):
        raise ValueError(
            f"Expected labels with shape {expected_labels_shape}, "
            f"got {tuple(labels.shape)}."
        )

    if labels.dtype != torch.long:
        raise TypeError(
            "labels must have dtype torch.long."
        )

    if labels.device != images.device:
        raise ValueError(
            "images and labels must be on the same device."
        )

    if not bool(torch.isfinite(images).all().item()):
        raise ValueError(
            "images contain NaN or infinite values."
        )

    image_min = float(
        images.detach().min().item()
    )
    image_max = float(
        images.detach().max().item()
    )

    if image_min < 0.0 or image_max > 1.0:
        raise ValueError(
            "DeepFool expects raw images in the [0, 1] pixel range. "
            f"Observed range: [{image_min:.6f}, {image_max:.6f}]."
        )

    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError(
            "max_steps must be a positive integer."
        )

    if overshoot < 0.0:
        raise ValueError(
            "overshoot must be non-negative."
        )

    if (
        not isinstance(num_classes, int)
        or num_classes < 2
    ):
        raise ValueError(
            "num_classes must be an integer of at least 2."
        )


def _deepfool_single(
    model: nn.Module,
    image: Tensor,
    max_steps: int,
    overshoot: float,
    num_classes: int,
) -> Tensor:
    """
    Runs untargeted L2 DeepFool on one image.

    DeepFool attacks the model's initial predicted class.
    Ground-truth labels are used later in evaluate.py to
    calculate robust accuracy and attack success rate.
    """

    original = image.detach().clone()

    # پیش‌بینی اولیه مدل، کلاس مبدأ حمله را مشخص می‌کند.
    with torch.no_grad():
        initial_logits = model(original)

    if (
        initial_logits.ndim != 2
        or initial_logits.shape[0] != 1
    ):
        raise ValueError(
            "The model must return logits with shape (N, K). "
            f"Received {tuple(initial_logits.shape)}."
        )

    model_num_classes = initial_logits.shape[1]

    if model_num_classes < 2:
        raise ValueError(
            "The model must predict at least two classes."
        )

    classes_to_consider = min(
        num_classes,
        model_num_classes,
    )

    original_class = int(
        initial_logits.argmax(dim=1).item()
    )

    # کلاس‌های دارای بیشترین logit بررسی می‌شوند.
    # برای CIFAR-10 و num_classes=10 تمام کلاس‌ها بررسی می‌شوند.
    candidate_classes = torch.topk(
        initial_logits[0],
        k=classes_to_consider,
        largest=True,
        sorted=True,
    ).indices.detach()

    total_perturbation = torch.zeros_like(original)
    adversarial = original.clone()

    # حتی اگر تابع از داخل torch.no_grad فراخوانی شده باشد،
    # برای ساخت حمله به گرادیان نیاز داریم.
    with torch.enable_grad():
        for _ in range(max_steps):
            adversarial = (
                adversarial
                .detach()
                .requires_grad_(True)
            )

            logits = model(adversarial)

            expected_shape = (
                1,
                model_num_classes,
            )

            if (
                logits.ndim != 2
                or tuple(logits.shape) != expected_shape
            ):
                raise ValueError(
                    "The model output shape changed during the attack. "
                    f"Expected {expected_shape}, "
                    f"got {tuple(logits.shape)}."
                )

            current_class = int(
                logits.argmax(dim=1).item()
            )

            # حمله زمانی موفق است که پیش‌بینی اولیه تغییر کند.
            if current_class != original_class:
                break

            original_logit = logits[
                0,
                original_class,
            ]

            # گرادیان logit کلاس اصلی نسبت به تصویر.
            original_gradient = torch.autograd.grad(
                outputs=original_logit,
                inputs=adversarial,
                retain_graph=True,
                create_graph=False,
                only_inputs=True,
            )[0]

            best_distance = math.inf

            best_direction: Optional[Tensor] = None
            best_direction_norm: Optional[Tensor] = None

            for class_index_tensor in candidate_classes:
                class_index = int(
                    class_index_tensor.item()
                )

                if class_index == original_class:
                    continue

                # گرادیان logit کلاس رقیب نسبت به تصویر.
                class_gradient = torch.autograd.grad(
                    outputs=logits[0, class_index],
                    inputs=adversarial,
                    retain_graph=True,
                    create_graph=False,
                    only_inputs=True,
                )[0]

                # بردار عمود بر مرز خطی‌شده‌ی دو کلاس:
                #
                # w_k = grad(f_k) - grad(f_original)
                boundary_normal = (
                    class_gradient
                    - original_gradient
                )

                boundary_normal_norm = (
                    boundary_normal
                    .flatten(1)
                    .norm(
                        p=2,
                        dim=1,
                    )[0]
                )

                norm_value = float(
                    boundary_normal_norm
                    .detach()
                    .item()
                )

                if (
                    not math.isfinite(norm_value)
                    or norm_value <= _MIN_GRADIENT_NORM
                ):
                    continue

                # اختلاف logit کلاس رقیب و کلاس اصلی.
                logit_difference = (
                    logits[0, class_index]
                    - original_logit
                )

                # فاصله تقریبی تصویر تا مرز تصمیم:
                #
                # |f_k - f_original| / ||w_k||_2
                distance = float(
                    (
                        logit_difference
                        .detach()
                        .abs()
                        / boundary_normal_norm.detach()
                    ).item()
                )

                if (
                    math.isfinite(distance)
                    and distance < best_distance
                ):
                    best_distance = distance
                    best_direction = (
                        boundary_normal.detach()
                    )
                    best_direction_norm = (
                        boundary_normal_norm.detach()
                    )

            # اگر گرادیان هیچ مرز معتبری پیدا نشد،
            # حمله برای این تصویر متوقف می‌شود.
            if (
                best_direction is None
                or best_direction_norm is None
            ):
                break

            # کمترین گام L2 برای رسیدن به نزدیک‌ترین مرز.
            step = (
                (best_distance + _BOUNDARY_OFFSET)
                * best_direction
                / best_direction_norm
            )

            total_perturbation = (
                total_perturbation
                + step
            )

            # overshoot نمونه را کمی از مرز عبور می‌دهد.
            adversarial = (
                original
                + (
                    1.0 + overshoot
                ) * total_perturbation
            ).clamp(
                0.0,
                1.0,
            ).detach()

    return adversarial


def deepfool_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    max_steps: int = 50,
    overshoot: float = 0.02,
    num_classes: int = 10,
) -> Tensor:
    """
    Generates untargeted L2 DeepFool adversarial examples.

    Parameters
    ----------
    model:
        Complete classifier containing Normalize and ResNet20.
        The model must accept raw images in [0, 1] and return logits.

    images:
        Float tensor with shape (N, 3, 32, 32)
        and values in the [0, 1] range.

    labels:
        Long tensor with shape (N,).

        DeepFool itself attacks the model's initial prediction.
        labels are kept for the common attack interface and are
        later used by evaluate.py to compute evaluation metrics.

    max_steps:
        Maximum number of DeepFool iterations per image.
        The project requirement is 50.

    overshoot:
        Extra multiplicative factor used to move the image
        slightly beyond the estimated decision boundary.

    num_classes:
        Number of highest-logit classes considered by DeepFool.
        Use 10 to consider all CIFAR-10 classes.

    Returns
    -------
    Tensor
        Adversarial images with the same shape, dtype and device
        as the input images. Values are clamped to [0, 1].

    Notes
    -----
    Images are processed sample-by-sample because different
    samples may cross different boundaries after different
    numbers of iterations.
    """

    _validate_inputs(
        images=images,
        labels=labels,
        max_steps=max_steps,
        overshoot=overshoot,
        num_classes=num_classes,
    )

    # وضعیت مدل و requires_grad پارامترها ذخیره می‌شود
    # تا بعد از حمله دقیقاً برگردانده شود.
    was_training = model.training

    parameter_requires_grad = [
        parameter.requires_grad
        for parameter in model.parameters()
    ]

    model.eval()

    try:
        # برای DeepFool تنها گرادیان ورودی لازم است.
        # گرادیان وزن‌های مدل محاسبه و ذخیره نمی‌شود.
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        adversarial_images = torch.empty_like(images)

        for sample_index in range(images.shape[0]):
            sample_slice = slice(
                sample_index,
                sample_index + 1,
            )

            adversarial_images[
                sample_slice
            ] = _deepfool_single(
                model=model,
                image=images[sample_slice],
                max_steps=max_steps,
                overshoot=overshoot,
                num_classes=num_classes,
            )

    finally:
        # وضعیت requires_grad پارامترها بازیابی می‌شود.
        for parameter, requires_grad in zip(
            model.parameters(),
            parameter_requires_grad,
        ):
            parameter.requires_grad_(
                requires_grad
            )

        # حالت train/eval قبلی مدل بازیابی می‌شود.
        model.train(was_training)

    return adversarial_images.detach()