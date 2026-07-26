from typing import Final

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


__all__ = ["cw_l2_attack"]


# برای جلوگیری از رسیدن ورودی atanh به دقیقاً -1 یا +1.
_TANH_EPSILON: Final[float] = 1e-6


def _validate_inputs(
    images: Tensor,
    labels: Tensor,
    c: float,
    kappa: float,
    learning_rate: float,
    steps: int,
) -> None:
    """
    Validates the common C&W L2 attack inputs.
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
            "C&W L2 expects raw images in the [0, 1] pixel range. "
            f"Observed range: [{image_min:.6f}, {image_max:.6f}]."
        )

    if (
        not isinstance(c, (int, float))
        or not math.isfinite(float(c))
        or c <= 0.0
    ):
        raise ValueError(
            "c must be a finite positive number."
        )

    if (
        not isinstance(kappa, (int, float))
        or not math.isfinite(float(kappa))
        or kappa < 0.0
    ):
        raise ValueError(
            "kappa must be a finite non-negative number."
        )

    if (
        not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0.0
    ):
        raise ValueError(
            "learning_rate must be a finite positive number."
        )

    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps <= 0
    ):
        raise ValueError(
            "steps must be a positive integer."
        )


def _to_tanh_space(images: Tensor) -> Tensor:
    """
    Maps images from [0, 1] to the unconstrained tanh space.

    Forward relation:
        x = 0.5 * (tanh(w) + 1)

    Therefore:
        w = atanh(2x - 1)
    """

    scaled = images * 2.0 - 1.0

    # atanh(-1) و atanh(+1) نامتناهی هستند.
    scaled = scaled.clamp(
        min=-1.0 + _TANH_EPSILON,
        max=1.0 - _TANH_EPSILON,
    )

    # atanh(x) = 0.5 * log((1+x)/(1-x))
    return 0.5 * torch.log(
        (1.0 + scaled) / (1.0 - scaled)
    )


def _from_tanh_space(w: Tensor) -> Tensor:
    """
    Maps an unconstrained tensor to valid images in [0, 1].
    """

    return 0.5 * (torch.tanh(w) + 1.0)


def _cw_margin_loss(
    logits: Tensor,
    labels: Tensor,
    kappa: float,
) -> tuple[Tensor, Tensor]:
    """
    Computes the untargeted C&W margin loss.

    For each sample:

        f(x') = max(
            Z(x')_y - max_{i != y} Z(x')_i,
            -kappa,
        )

    Parameters
    ----------
    logits:
        Model logits with shape (N, K).

    labels:
        Ground-truth labels with shape (N,).

    kappa:
        Required confidence margin.

    Returns
    -------
    loss_per_sample:
        C&W classification loss for every sample.

    raw_margin:
        true_logit - largest_non_true_logit before clamping.
    """

    if logits.ndim != 2:
        raise ValueError(
            "The model must return logits with shape (N, K), "
            f"got {tuple(logits.shape)}."
        )

    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "The logits batch size must match the labels batch size."
        )

    num_classes = logits.shape[1]

    if num_classes < 2:
        raise ValueError(
            "The model must predict at least two classes."
        )

    invalid_labels = (
        (labels < 0)
        | (labels >= num_classes)
    )

    if bool(invalid_labels.any().item()):
        raise ValueError(
            "labels contain a class index outside "
            "the model output range."
        )

    # Z(x')_y
    true_logits = logits.gather(
        dim=1,
        index=labels.unsqueeze(1),
    ).squeeze(1)

    # کلاس واقعی را ماسک می‌کنیم تا بتوانیم
    # بیشترین logit بین سایر کلاس‌ها را پیدا کنیم.
    true_class_mask = F.one_hot(
        labels,
        num_classes=num_classes,
    ).bool()

    largest_other_logits = logits.masked_fill(
        true_class_mask,
        -torch.inf,
    ).max(
        dim=1,
    ).values

    # Z_y - max_{i != y}(Z_i)
    raw_margin = (
        true_logits
        - largest_other_logits
    )

    # max(raw_margin, -kappa)
    loss_per_sample = torch.clamp(
        raw_margin,
        min=-float(kappa),
    )

    return loss_per_sample, raw_margin


def _successful_untargeted_attack(
    logits: Tensor,
    labels: Tensor,
    raw_margin: Tensor,
    kappa: float,
) -> Tensor:
    """
    Returns a boolean success mask for untargeted C&W L2.

    A sample is successful when:

        1. predicted_label != true_label

        2. true_logit - max_other_logit <= -kappa
    """

    predictions = logits.argmax(dim=1)

    changed_class = predictions.ne(labels)

    confidence_satisfied = raw_margin.le(
        -float(kappa)
    )

    return (
        changed_class
        & confidence_satisfied
    )


def cw_l2_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    c: float = 1.0,
    kappa: float = 0.0,
    learning_rate: float = 0.01,
    steps: int = 100,
) -> Tensor:
    """
    Generates untargeted C&W L2 adversarial examples.

    Parameters
    ----------
    model:
        Complete classifier containing Normalize and ResNet20.

        The model must accept raw images in [0, 1]
        and return logits with shape (N, K).

    images:
        Floating-point tensor with shape (N, C, H, W)
        and values in the [0, 1] range.

    labels:
        Ground-truth class indices with shape (N,)
        and dtype torch.long.

    c:
        Weight of the C&W classification term.

        Larger values usually prioritize attack success more strongly,
        but may produce larger perturbations.

    kappa:
        Required attack confidence margin.

        The project baseline uses:
            kappa = 0.0

    learning_rate:
        Adam learning rate for optimizing the auxiliary variable w.

    steps:
        Number of Adam iterations.

        The project requirement is:
            steps = 100

    Returns
    -------
    Tensor
        Adversarial images with the same shape, dtype and device
        as images.

        All returned pixel values are in [0, 1].

    Notes
    -----
    The optimization objective is:

        ||x_adv - x||_2^2 + c * f(x_adv)

    where:

        f(x_adv) = max(
            Z(x_adv)_y - max_{i != y} Z(x_adv)_i,
            -kappa,
        )

    Instead of optimizing x_adv directly, the attack optimizes w:

        x_adv = 0.5 * (tanh(w) + 1)

    Therefore, x_adv always remains inside [0, 1].
    """

    _validate_inputs(
        images=images,
        labels=labels,
        c=c,
        kappa=kappa,
        learning_rate=learning_rate,
        steps=steps,
    )

    original_images = (
        images
        .detach()
        .clone()
    )

    labels = labels.detach()

    # وضعیت اولیه مدل ذخیره می‌شود.
    was_training = model.training

    # وضعیت requires_grad تمام پارامترها ذخیره می‌شود.
    parameter_requires_grad = [
        parameter.requires_grad
        for parameter in model.parameters()
    ]

    model.eval()

    try:
        # در C&W فقط نسبت به w گرادیان لازم داریم.
        # گرادیان وزن‌های مدل نباید محاسبه شود.
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        # تصویر اصلی به فضای بدون محدودیت w منتقل می‌شود.
        initial_w = _to_tanh_space(
            original_images
        ).detach()

        w = nn.Parameter(
            initial_w.clone()
        )

        optimizer = torch.optim.Adam(
            [w],
            lr=float(learning_rate),
        )

        batch_size = original_images.shape[0]

        # کمترین L2 موفق برای هر نمونه.
        best_l2_squared = torch.full(
            size=(batch_size,),
            fill_value=torch.inf,
            device=images.device,
            dtype=images.dtype,
        )

        # بهترین تصویر موفق هر نمونه.
        best_adversarial = (
            original_images.clone()
        )

        # مشخص می‌کند آیا تاکنون برای هر نمونه
        # یک خروجی موفق پیدا شده است یا خیر.
        success_found = torch.zeros(
            batch_size,
            device=images.device,
            dtype=torch.bool,
        )

        for _ in range(steps):
            optimizer.zero_grad(
                set_to_none=True
            )

            # این تبدیل تضمین می‌کند خروجی در [0,1] باشد.
            adversarial = _from_tanh_space(w)

            logits = model(adversarial)

            (
                classification_loss,
                raw_margin,
            ) = _cw_margin_loss(
                logits=logits,
                labels=labels,
                kappa=kappa,
            )

            # ||x_adv - x||_2^2
            l2_squared = (
                adversarial
                - original_images
            ).flatten(1).pow(2).sum(dim=1)

            # مجموع تابع هدف تمام نمونه‌های batch.
            total_loss = (
                l2_squared
                + float(c) * classification_loss
            ).sum()

            if not bool(
                torch.isfinite(total_loss).item()
            ):
                raise RuntimeError(
                    "C&W optimization produced "
                    "a non-finite loss."
                )

            # بهترین نمونه موفق بر اساس کمترین L2 ذخیره می‌شود.
            with torch.no_grad():
                successful = (
                    _successful_untargeted_attack(
                        logits=logits,
                        labels=labels,
                        raw_margin=raw_margin,
                        kappa=kappa,
                    )
                )

                improved = (
                    successful
                    & l2_squared.lt(
                        best_l2_squared
                    )
                )

                if bool(improved.any().item()):
                    best_l2_squared[improved] = (
                        l2_squared[improved]
                    )

                    best_adversarial[improved] = (
                        adversarial[improved]
                    )

                    success_found[improved] = True

            total_loss.backward()
            optimizer.step()

        # آخرین optimizer.step پس از آخرین forward اجرا شده است؛
        # بنابراین خروجی نهایی یک بار دیگر بررسی می‌شود.
        with torch.no_grad():
            final_adversarial = (
                _from_tanh_space(w)
            )

            final_logits = model(
                final_adversarial
            )

            (
                _,
                final_raw_margin,
            ) = _cw_margin_loss(
                logits=final_logits,
                labels=labels,
                kappa=kappa,
            )

            final_l2_squared = (
                final_adversarial
                - original_images
            ).flatten(1).pow(2).sum(dim=1)

            final_successful = (
                _successful_untargeted_attack(
                    logits=final_logits,
                    labels=labels,
                    raw_margin=final_raw_margin,
                    kappa=kappa,
                )
            )

            final_improved = (
                final_successful
                & final_l2_squared.lt(
                    best_l2_squared
                )
            )

            if bool(
                final_improved.any().item()
            ):
                best_l2_squared[
                    final_improved
                ] = final_l2_squared[
                    final_improved
                ]

                best_adversarial[
                    final_improved
                ] = final_adversarial[
                    final_improved
                ]

                success_found[
                    final_improved
                ] = True

            # اگر حمله برای یک نمونه موفق بوده،
            # بهترین نمونه موفق با کمترین L2 برمی‌گردد.
            #
            # اگر موفق نبوده، آخرین تصویر بهینه‌شده برمی‌گردد.
            result = torch.where(
                success_found.view(
                    -1,
                    1,
                    1,
                    1,
                ),
                best_adversarial,
                final_adversarial,
            )

    finally:
        # وضعیت requires_grad پارامترهای مدل بازیابی می‌شود.
        for parameter, requires_grad in zip(
            model.parameters(),
            parameter_requires_grad,
        ):
            parameter.requires_grad_(
                requires_grad
            )

        # حالت train/eval اولیه مدل بازیابی می‌شود.
        model.train(was_training)

    result = result.detach()

    if not bool(
        torch.isfinite(result).all().item()
    ):
        raise RuntimeError(
            "C&W L2 produced NaN or "
            "infinite pixel values."
        )

    return result.clamp(
        0.0,
        1.0,
    )