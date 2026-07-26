from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

import math

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


__all__ = [
    "add_gaussian_noise",
    "gaussian_fine_tune",
    "predict_smoothed",
]


def _model_device(
    model: nn.Module,
) -> torch.device:
    """
    Returns the device on which the model is located.
    """

    first_parameter = next(
        model.parameters(),
        None,
    )

    if first_parameter is not None:
        return first_parameter.device

    first_buffer = next(
        model.buffers(),
        None,
    )

    if first_buffer is not None:
        return first_buffer.device

    return torch.device("cpu")


def _validate_images(
    images: Tensor,
) -> None:
    """
    Validates clean input images.

    Clean images must be raw CIFAR-10 images in [0, 1].
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

    if not bool(
        torch.isfinite(images).all().item()
    ):
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
            "Clean images must be in the raw [0, 1] pixel range. "
            f"Observed range: [{image_min:.6f}, {image_max:.6f}]."
        )


def _validate_labels(
    labels: Tensor,
    batch_size: int,
) -> None:
    """
    Validates classification labels.
    """

    if (
        labels.ndim != 1
        or labels.shape[0] != batch_size
    ):
        raise ValueError(
            "Expected labels with shape (N,), "
            "matching the image batch."
        )

    if labels.dtype != torch.long:
        raise TypeError(
            "labels must have dtype torch.long."
        )


def _validate_positive_number(
    value: float,
    name: str,
) -> float:
    """
    Validates positive numeric hyperparameters.
    """

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(
            f"{name} must be a finite positive number."
        )

    return float(value)


def _add_noise(
    images: Tensor,
    sigma: float,
    clip_noise: bool,
    generator: Optional[torch.Generator],
) -> Tensor:
    """
    Internal Gaussian-noise function without input validation.
    """

    noise = torch.randn(
        images.shape,
        device=images.device,
        dtype=images.dtype,
        generator=generator,
    )

    noise = noise * sigma

    noisy_images = images + noise

    if clip_noise:
        noisy_images = noisy_images.clamp(
            0.0,
            1.0,
        )

    return noisy_images


def add_gaussian_noise(
    images: Tensor,
    sigma: float = 0.25,
    clip_noise: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Adds Gaussian noise to clean images in raw pixel space.

    The randomized-smoothing distribution is:

        x_noisy = x + epsilon

        epsilon ~ N(0, sigma^2 I)

    Parameters
    ----------
    images:
        Clean images with shape (N, C, H, W)
        and values in [0, 1].

    sigma:
        Gaussian standard deviation.

        Project default:
            sigma = 0.25

    clip_noise:
        If True, noisy images are clamped to [0, 1].

        The default is False because the standard randomized-smoothing
        formulation uses the unmodified Gaussian distribution x + epsilon.

        If clipping is enabled during fine-tuning, it must also be enabled
        during smoothed inference so that the two distributions match.

    generator:
        Optional torch.Generator for reproducible noise.

    Returns
    -------
    Tensor:
        Gaussian-noised images with the same shape, dtype and device
        as the input images.
    """

    _validate_images(
        images
    )

    sigma = _validate_positive_number(
        sigma,
        "sigma",
    )

    if not isinstance(
        clip_noise,
        bool,
    ):
        raise TypeError(
            "clip_noise must be a boolean."
        )

    return _add_noise(
        images=images,
        sigma=sigma,
        clip_noise=clip_noise,
        generator=generator,
    )


def _evaluate_accuracy(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    sigma: Optional[float],
    clip_noise: bool,
) -> float:
    """
    Evaluates either clean or single-noise-sample accuracy.

    This function is used for selecting the best Gaussian-fine-tuned
    checkpoint in memory.
    """

    previous_mode = model.training

    model.eval()

    total = 0
    correct = 0
    first_batch = True

    try:
        with torch.inference_mode():
            for batch in data_loader:
                if (
                    not isinstance(
                        batch,
                        (tuple, list),
                    )
                    or len(batch) < 2
                ):
                    raise ValueError(
                        "Each data-loader batch must contain "
                        "images and labels."
                    )

                images = batch[0].to(
                    device,
                    non_blocking=True,
                )

                labels = batch[1].to(
                    device,
                    non_blocking=True,
                )

                if first_batch:
                    _validate_images(
                        images
                    )
                    first_batch = False

                _validate_labels(
                    labels=labels,
                    batch_size=images.shape[0],
                )

                if sigma is not None:
                    images = _add_noise(
                        images=images,
                        sigma=sigma,
                        clip_noise=clip_noise,
                        generator=None,
                    )

                logits = model(
                    images
                )

                if (
                    logits.ndim != 2
                    or logits.shape[0]
                    != images.shape[0]
                ):
                    raise ValueError(
                        "The model must return logits "
                        "with shape (N, K)."
                    )

                predictions = logits.argmax(
                    dim=1
                )

                correct += int(
                    predictions
                    .eq(labels)
                    .sum()
                    .item()
                )

                total += int(
                    labels.numel()
                )

    finally:
        model.train(
            previous_mode
        )

    if total == 0:
        raise ValueError(
            "The data loader produced no samples."
        )

    return correct / total


def gaussian_fine_tune(
    model: nn.Module,
    train_loader: Iterable,
    optimizer: Optimizer,
    epochs: int = 5,
    sigma: float = 0.25,
    device: Optional[
        torch.device | str
    ] = None,
    criterion: Optional[nn.Module] = None,
    validation_loader: Optional[Iterable] = None,
    clip_noise: bool = False,
    restore_best: bool = True,
) -> dict[str, Any]:
    """
    Fine-tunes the classifier using Gaussian-noised images.

    At every training step:

        1. A new Gaussian-noised version of the batch is generated.
        2. The noisy batch is passed to the model.
        3. Cross-entropy loss is computed.
        4. Model parameters are updated.

    This function changes the model weights but performs no disk I/O.

    If validation_loader is provided, the best state is selected using
    noisy validation accuracy. Otherwise, the final epoch is returned.

    Parameters
    ----------
    model:
        Complete model containing Normalize and ResNet20.

    train_loader:
        CIFAR-10 training loader.

        Images must only use transforms.ToTensor().
        Do not Normalize images outside the model.

    optimizer:
        Optimizer created from model parameters.

    epochs:
        Number of Gaussian fine-tuning epochs.

    sigma:
        Gaussian noise standard deviation.

        Project default:
            sigma = 0.25

    device:
        Model device.

        The model must be moved to this device before the optimizer
        is created.

    criterion:
        Classification loss.

        Default:
            nn.CrossEntropyLoss()

    validation_loader:
        Optional validation subset taken from the CIFAR-10 training split.

        The test split must not be used for checkpoint selection.

    clip_noise:
        Whether Gaussian-noised images are clamped to [0, 1].

        The same value must later be passed to predict_smoothed().

    restore_best:
        If True, the best in-memory state is loaded into model after
        training finishes.

    Returns
    -------
    Dictionary containing:

        history
        best_epoch
        best_metric
        best_state_dict
        selection_metric
        sigma
        epochs
        clip_noise
    """

    sigma = _validate_positive_number(
        sigma,
        "sigma",
    )

    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs <= 0
    ):
        raise ValueError(
            "epochs must be a positive integer."
        )

    if not isinstance(
        optimizer,
        Optimizer,
    ):
        raise TypeError(
            "optimizer must be a torch.optim.Optimizer."
        )

    if not isinstance(
        clip_noise,
        bool,
    ):
        raise TypeError(
            "clip_noise must be a boolean."
        )

    if not isinstance(
        restore_best,
        bool,
    ):
        raise TypeError(
            "restore_best must be a boolean."
        )

    current_device = _model_device(
        model
    )

    if device is None:
        resolved_device = current_device

    else:
        resolved_device = torch.device(
            device
        )

        if resolved_device != current_device:
            raise ValueError(
                "Move the model to the desired device before "
                "creating the optimizer."
            )

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_clean_accuracy": [],
        "val_noisy_accuracy": [],
    }

    best_state_dict: Optional[
        dict[str, Tensor]
    ] = None

    best_epoch = -1
    best_metric = -math.inf

    for epoch in range(epochs):
        model.train()

        total = 0
        correct = 0
        total_loss = 0.0
        first_batch = True

        for batch in train_loader:
            if (
                not isinstance(
                    batch,
                    (tuple, list),
                )
                or len(batch) < 2
            ):
                raise ValueError(
                    "Each train-loader batch must contain "
                    "images and labels."
                )

            images = batch[0].to(
                resolved_device,
                non_blocking=True,
            )

            labels = batch[1].to(
                resolved_device,
                non_blocking=True,
            )

            if first_batch:
                _validate_images(
                    images
                )
                first_batch = False

            _validate_labels(
                labels=labels,
                batch_size=images.shape[0],
            )

            noisy_images = _add_noise(
                images=images,
                sigma=sigma,
                clip_noise=clip_noise,
                generator=None,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                noisy_images
            )

            if (
                logits.ndim != 2
                or logits.shape[0]
                != images.shape[0]
            ):
                raise ValueError(
                    "The model must return logits "
                    "with shape (N, K)."
                )

            loss = criterion(
                logits,
                labels,
            )

            if loss.ndim != 0:
                raise ValueError(
                    "criterion must return a scalar loss."
                )

            if not bool(
                torch.isfinite(loss).item()
            ):
                raise RuntimeError(
                    "Gaussian fine-tuning produced "
                    "a non-finite loss."
                )

            loss.backward()
            optimizer.step()

            batch_size = int(
                labels.numel()
            )

            total_loss += (
                float(
                    loss.detach().item()
                )
                * batch_size
            )

            correct += int(
                logits
                .detach()
                .argmax(dim=1)
                .eq(labels)
                .sum()
                .item()
            )

            total += batch_size

        if total == 0:
            raise ValueError(
                "The train loader produced no samples."
            )

        train_loss = (
            total_loss / total
        )

        train_accuracy = (
            correct / total
        )

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        if validation_loader is not None:
            val_clean_accuracy = (
                _evaluate_accuracy(
                    model=model,
                    data_loader=validation_loader,
                    device=resolved_device,
                    sigma=None,
                    clip_noise=clip_noise,
                )
            )

            val_noisy_accuracy = (
                _evaluate_accuracy(
                    model=model,
                    data_loader=validation_loader,
                    device=resolved_device,
                    sigma=sigma,
                    clip_noise=clip_noise,
                )
            )

            history[
                "val_clean_accuracy"
            ].append(
                val_clean_accuracy
            )

            history[
                "val_noisy_accuracy"
            ].append(
                val_noisy_accuracy
            )

            selection_metric = (
                val_noisy_accuracy
            )

            should_store = (
                selection_metric
                > best_metric
            )

        else:
            selection_metric = (
                train_accuracy
            )

            # Without validation data, the final epoch is retained.
            should_store = (
                epoch == epochs - 1
            )

        if should_store:
            best_metric = (
                selection_metric
            )

            best_epoch = epoch

            # State is copied to CPU so it does not occupy GPU memory.
            best_state_dict = {
                key: value
                .detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

    if best_state_dict is None:
        raise RuntimeError(
            "No model state was produced."
        )

    if restore_best:
        model.load_state_dict(
            best_state_dict
        )

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_state_dict": best_state_dict,
        "selection_metric": (
            "val_noisy_accuracy"
            if validation_loader is not None
            else "final_train_accuracy"
        ),
        "sigma": sigma,
        "epochs": epochs,
        "clip_noise": clip_noise,
    }


def predict_smoothed(
    model: nn.Module,
    images: Tensor,
    sigma: float = 0.25,
    num_samples: int = 100,
    noise_batch_size: int = 25,
    num_classes: int = 10,
    clip_noise: bool = False,
    generator: Optional[
        torch.Generator
    ] = None,
) -> tuple[Tensor, Tensor]:
    """
    Predicts classes using randomized smoothing.

    For every input image:

        1. Generate num_samples Gaussian-noised copies.
        2. Run the classifier on all noisy copies.
        3. Count predicted classes.
        4. Return the class with the highest vote count.

    Project defaults:

        sigma = 0.25
        num_samples = 100

    Parameters
    ----------
    model:
        Complete model containing Normalize and ResNet20.

    images:
        Clean or adversarial raw images in [0, 1].

    sigma:
        Gaussian standard deviation.

    num_samples:
        Number of noisy copies per image.

    noise_batch_size:
        Number of noisy copies processed per image in each forward step.

        The real model batch size passed into the classifier is:

            input_batch_size * noise_batch_size

        Reduce this value if GPU memory is insufficient.

    num_classes:
        Number of classes.

        CIFAR-10:
            num_classes = 10

    clip_noise:
        Must match the value used during Gaussian fine-tuning.

    generator:
        Optional random generator.

    Returns
    -------
    predictions:
        Tensor with shape (N,) and dtype torch.long.

    vote_counts:
        Tensor with shape (N, num_classes) and dtype torch.long.
    """

    _validate_images(
        images
    )

    sigma = _validate_positive_number(
        sigma,
        "sigma",
    )

    if (
        isinstance(num_samples, bool)
        or not isinstance(num_samples, int)
        or num_samples <= 0
    ):
        raise ValueError(
            "num_samples must be a positive integer."
        )

    if (
        isinstance(noise_batch_size, bool)
        or not isinstance(noise_batch_size, int)
        or noise_batch_size <= 0
    ):
        raise ValueError(
            "noise_batch_size must be a positive integer."
        )

    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes < 2
    ):
        raise ValueError(
            "num_classes must be an integer "
            "of at least 2."
        )

    if not isinstance(
        clip_noise,
        bool,
    ):
        raise TypeError(
            "clip_noise must be a boolean."
        )

    if images.device != _model_device(model):
        raise ValueError(
            "images and model must be "
            "on the same device."
        )

    previous_mode = model.training

    model.eval()

    (
        batch_size,
        channels,
        height,
        width,
    ) = images.shape

    vote_counts = torch.zeros(
        size=(
            batch_size,
            num_classes,
        ),
        device=images.device,
        dtype=torch.long,
    )

    generated_samples = 0

    try:
        with torch.inference_mode():
            while generated_samples < num_samples:
                current_count = min(
                    noise_batch_size,
                    num_samples
                    - generated_samples,
                )

                # Shape:
                # (N, current_count, C, H, W)
                expanded_images = (
                    images
                    .unsqueeze(1)
                    .expand(
                        batch_size,
                        current_count,
                        channels,
                        height,
                        width,
                    )
                )

                noise = torch.randn(
                    expanded_images.shape,
                    device=images.device,
                    dtype=images.dtype,
                    generator=generator,
                )

                noise = noise * sigma

                noisy_images = (
                    expanded_images
                    + noise
                )

                if clip_noise:
                    noisy_images = (
                        noisy_images.clamp(
                            0.0,
                            1.0,
                        )
                    )

                # Shape passed to classifier:
                # (N * current_count, C, H, W)
                noisy_images = (
                    noisy_images.reshape(
                        batch_size
                        * current_count,
                        channels,
                        height,
                        width,
                    )
                )

                logits = model(
                    noisy_images
                )

                expected_shape = (
                    batch_size
                    * current_count,
                    num_classes,
                )

                if (
                    logits.ndim != 2
                    or tuple(logits.shape)
                    != expected_shape
                ):
                    raise ValueError(
                        "Expected model output shape "
                        f"{expected_shape}, got "
                        f"{tuple(logits.shape)}."
                    )

                noisy_predictions = (
                    logits
                    .argmax(dim=1)
                    .reshape(
                        batch_size,
                        current_count,
                    )
                )

                chunk_votes = (
                    torch
                    .nn
                    .functional
                    .one_hot(
                        noisy_predictions,
                        num_classes=num_classes,
                    )
                    .sum(dim=1)
                    .to(dtype=torch.long)
                )

                vote_counts += (
                    chunk_votes
                )

                generated_samples += (
                    current_count
                )

    finally:
        model.train(
            previous_mode
        )

    # Every image must receive exactly num_samples votes.
    vote_totals = vote_counts.sum(
        dim=1
    )

    if not bool(
        torch.all(
            vote_totals == num_samples
        ).item()
    ):
        raise RuntimeError(
            "The number of collected votes is incorrect."
        )

    predictions = vote_counts.argmax(
        dim=1
    )

    return predictions, vote_counts