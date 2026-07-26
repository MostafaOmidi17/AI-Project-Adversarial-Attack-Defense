from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import torch
from torch import Tensor, nn


__all__ = [
    "DEFAULT_DIFFUSION_MODEL_ID",
    "load_diffusion_components",
    "purify_images",
]


DEFAULT_DIFFUSION_MODEL_ID = "google/ddpm-cifar10-32"


def _config_value(
    config: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Reads a value from a Diffusers config object or mapping.
    """

    value = getattr(config, key, default)

    if value is default and isinstance(config, Mapping):
        value = config.get(key, default)

    return value


def _module_device_and_dtype(
    module: nn.Module,
) -> tuple[torch.device, torch.dtype]:
    """
    Returns the device and floating-point dtype of a module.
    """

    parameter = next(
        module.parameters(),
        None,
    )

    if parameter is not None:
        return parameter.device, parameter.dtype

    buffer = next(
        module.buffers(),
        None,
    )

    if buffer is not None:
        return buffer.device, buffer.dtype

    return torch.device("cpu"), torch.float32


def _validate_images(
    images: Tensor,
) -> None:
    """
    Validates raw CIFAR-10 images in the [0, 1] pixel range.
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
            "Diffusion purification expects raw images "
            "in the [0, 1] pixel range. "
            f"Observed range: [{image_min:.6f}, {image_max:.6f}]."
        )


def _validate_components(
    images: Tensor,
    unet: nn.Module,
    scheduler: Any,
    timestep: int,
) -> tuple[torch.device, torch.dtype, int]:
    """
    Validates the UNet, scheduler, image shape, and timestep.
    """

    unet_device, unet_dtype = (
        _module_device_and_dtype(
            unet
        )
    )

    if images.device != unet_device:
        raise ValueError(
            "images and the diffusion UNet must be "
            "on the same device. "
            f"Images: {images.device}; "
            f"UNet: {unet_device}."
        )

    if not hasattr(unet, "config"):
        raise TypeError(
            "unet must expose a Diffusers-style config."
        )

    if not hasattr(scheduler, "config"):
        raise TypeError(
            "scheduler must expose a Diffusers-style config."
        )

    required_scheduler_methods = (
        "add_noise",
        "scale_model_input",
        "set_timesteps",
        "step",
    )

    for method_name in required_scheduler_methods:
        if not callable(
            getattr(
                scheduler,
                method_name,
                None,
            )
        ):
            raise TypeError(
                "scheduler must provide a callable "
                f"{method_name}() method."
            )

    expected_channels = _config_value(
        unet.config,
        "in_channels",
        None,
    )

    if (
        expected_channels is not None
        and images.shape[1]
        != int(expected_channels)
    ):
        raise ValueError(
            "The diffusion UNet expects "
            f"{expected_channels} channels, "
            f"but images have {images.shape[1]}."
        )

    expected_size = _config_value(
        unet.config,
        "sample_size",
        None,
    )

    if isinstance(expected_size, int):
        expected_height = expected_size
        expected_width = expected_size

    elif (
        isinstance(
            expected_size,
            (tuple, list),
        )
        and len(expected_size) == 2
    ):
        expected_height = int(
            expected_size[0]
        )

        expected_width = int(
            expected_size[1]
        )

    else:
        expected_height = None
        expected_width = None

    if (
        expected_height is not None
        and (
            images.shape[-2]
            != expected_height
            or images.shape[-1]
            != expected_width
        )
    ):
        raise ValueError(
            "Image spatial size does not match "
            "the diffusion UNet. "
            f"Expected ({expected_height}, {expected_width}), "
            f"got {tuple(images.shape[-2:])}."
        )

    num_train_timesteps = _config_value(
        scheduler.config,
        "num_train_timesteps",
        None,
    )

    if num_train_timesteps is None:
        raise ValueError(
            "scheduler.config.num_train_timesteps "
            "is required."
        )

    num_train_timesteps = int(
        num_train_timesteps
    )

    if (
        isinstance(timestep, bool)
        or not isinstance(timestep, int)
        or timestep < 0
        or timestep >= num_train_timesteps
    ):
        raise ValueError(
            "timestep must be an integer in "
            f"[0, {num_train_timesteps - 1}], "
            f"got {timestep}."
        )

    if not unet_dtype.is_floating_point:
        raise TypeError(
            "The diffusion UNet must use "
            "a floating-point dtype."
        )

    return (
        unet_device,
        unet_dtype,
        num_train_timesteps,
    )


def _validate_generator(
    generator: Optional[torch.Generator],
    device: torch.device,
) -> None:
    """
    Checks that an optional random generator matches
    the model device.
    """

    if generator is None:
        return

    generator_device = getattr(
        generator,
        "device",
        None,
    )

    if generator_device is None:
        return

    generator_device = torch.device(
        generator_device
    )

    if generator_device.type != device.type:
        raise ValueError(
            "generator and the diffusion UNet must "
            "use the same device type. "
            f"Generator: {generator_device}; "
            f"UNet: {device}."
        )


def _extract_model_sample(
    model_output: Any,
    expected_shape: torch.Size,
) -> Tensor:
    """
    Extracts the predicted-noise tensor from a UNet output.
    """

    if hasattr(
        model_output,
        "sample",
    ):
        sample = model_output.sample

    elif (
        isinstance(
            model_output,
            (tuple, list),
        )
        and len(model_output) > 0
    ):
        sample = model_output[0]

    else:
        raise TypeError(
            "The diffusion UNet output must expose "
            ".sample or be a non-empty tuple."
        )

    if not torch.is_tensor(sample):
        raise TypeError(
            "The diffusion UNet prediction "
            "must be a tensor."
        )

    if sample.shape != expected_shape:
        raise ValueError(
            "The diffusion UNet output shape "
            "must match its input shape. "
            f"Expected {tuple(expected_shape)}, "
            f"got {tuple(sample.shape)}."
        )

    return sample


def _prepare_reverse_timesteps(
    scheduler: Any,
    timestep: int,
    num_train_timesteps: int,
    device: torch.device,
) -> Tensor:
    """
    Creates the exact consecutive reverse chain:

        timestep, timestep - 1, ..., 0

    The scheduler is configured with all training timesteps
    so every reverse update corresponds to exactly one DDPM step.
    """

    # مهم:
    # اگر مستقیماً num_inference_steps=timestep+1 قرار دهیم،
    # scheduler ممکن است timestepها را در کل بازه 0 تا 999 پخش کند.
    #
    # ما به زنجیره دقیق زیر نیاز داریم:
    #
    # 50, 49, 48, ..., 1, 0
    #
    # بنابراین ابتدا تمام timestepهای آموزشی را فعال می‌کنیم
    # و سپس بخش موردنیاز را جدا می‌کنیم.
    scheduler.set_timesteps(
        num_inference_steps=num_train_timesteps,
        device=device,
    )

    scheduler_timesteps = (
        scheduler.timesteps.to(
            device=device
        )
    )

    reverse_timesteps = scheduler_timesteps[
        scheduler_timesteps <= timestep
    ]

    expected_timesteps = torch.arange(
        timestep,
        -1,
        -1,
        device=device,
        dtype=reverse_timesteps.dtype,
    )

    if (
        reverse_timesteps.shape
        != expected_timesteps.shape
        or not bool(
            torch.equal(
                reverse_timesteps,
                expected_timesteps,
            )
        )
    ):
        raise RuntimeError(
            "The scheduler did not produce "
            "the required consecutive reverse timesteps "
            f"from {timestep} to 0."
        )

    return reverse_timesteps


def load_diffusion_components(
    model_id: str = DEFAULT_DIFFUSION_MODEL_ID,
    device: Optional[
        torch.device | str
    ] = None,
    dtype: Optional[
        torch.dtype
    ] = None,
    local_files_only: bool = False,
) -> tuple[nn.Module, Any]:
    """
    Loads the pretrained CIFAR-10 diffusion UNet
    and its matching DDPM scheduler.

    This function only loads the diffusion components.
    It does not purify images, classify images, or save files.

    Parameters
    ----------
    model_id:
        Hugging Face model identifier.

        Project model:
            google/ddpm-cifar10-32

    device:
        Device used by the diffusion UNet.

    dtype:
        Floating-point dtype used by the diffusion UNet.

        Recommended default:
            torch.float32

    local_files_only:
        If True, the model must already exist in the local
        Hugging Face cache.

    Returns
    -------
    unet:
        Pretrained UNet2DModel.

    scheduler:
        Matching DDPMScheduler.
    """

    if (
        not isinstance(model_id, str)
        or not model_id.strip()
    ):
        raise ValueError(
            "model_id must be a non-empty string."
        )

    if not isinstance(
        local_files_only,
        bool,
    ):
        raise TypeError(
            "local_files_only must be a boolean."
        )

    try:
        from diffusers import (
            DDPMScheduler,
            UNet2DModel,
        )

    except ImportError as error:
        raise ImportError(
            "diffusers is required for diffusion purification. "
            "Install it with:\n"
            "pip install -U diffusers "
            "huggingface_hub safetensors accelerate"
        ) from error

    if device is None:
        resolved_device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:
        resolved_device = torch.device(
            device
        )

    if dtype is None:
        resolved_dtype = torch.float32

    else:
        resolved_dtype = dtype

    supported_dtypes = (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    )

    if resolved_dtype not in supported_dtypes:
        raise ValueError(
            "dtype must be torch.float32, "
            "torch.float16, or torch.bfloat16."
        )

    if (
        resolved_device.type == "cpu"
        and resolved_dtype == torch.float16
    ):
        raise ValueError(
            "torch.float16 is not supported reliably "
            "for this UNet on CPU. "
            "Use torch.float32 on CPU."
        )

    unet = UNet2DModel.from_pretrained(
        model_id,
        torch_dtype=resolved_dtype,
        use_safetensors=True,
        local_files_only=local_files_only,
    )

    scheduler = DDPMScheduler.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )

    unet = unet.to(
        device=resolved_device,
        dtype=resolved_dtype,
    )

    unet.eval()

    # مدل diffusion فقط برای inference استفاده می‌شود.
    unet.requires_grad_(False)

    return unet, scheduler


def _purify_batch(
    images: Tensor,
    unet: nn.Module,
    scheduler: Any,
    timestep: int,
    generator: Optional[torch.Generator],
) -> Tensor:
    """
    Purifies one image batch using forward noise
    followed by reverse DDPM.
    """

    (
        device,
        unet_dtype,
        num_train_timesteps,
    ) = _validate_components(
        images=images,
        unet=unet,
        scheduler=scheduler,
        timestep=timestep,
    )

    _validate_generator(
        generator=generator,
        device=device,
    )

    input_dtype = images.dtype

    with torch.inference_mode():
        # مدل DDPM روی داده‌های تصویری در فضای [-1, 1]
        # کار می‌کند.
        clean_sample = images.to(
            dtype=unet_dtype
        )

        clean_sample = (
            clean_sample * 2.0 - 1.0
        )

        forward_noise = torch.randn(
            clean_sample.shape,
            device=device,
            dtype=unet_dtype,
            generator=generator,
        )

        batch_timesteps = torch.full(
            size=(images.shape[0],),
            fill_value=timestep,
            device=device,
            dtype=torch.long,
        )

        # Forward process:
        #
        # x_t =
        # sqrt(alpha_bar_t) * x_0
        # +
        # sqrt(1 - alpha_bar_t) * epsilon
        current_sample = scheduler.add_noise(
            original_samples=clean_sample,
            noise=forward_noise,
            timesteps=batch_timesteps,
        )

        reverse_timesteps = (
            _prepare_reverse_timesteps(
                scheduler=scheduler,
                timestep=timestep,
                num_train_timesteps=(
                    num_train_timesteps
                ),
                device=device,
            )
        )

        # Reverse process:
        #
        # timestep, timestep-1, ..., 1, 0
        for current_timestep in reverse_timesteps:
            model_input = (
                scheduler.scale_model_input(
                    current_sample,
                    current_timestep,
                )
            )

            unet_output = unet(
                model_input,
                current_timestep,
            )

            predicted_noise = (
                _extract_model_sample(
                    model_output=unet_output,
                    expected_shape=(
                        current_sample.shape
                    ),
                )
            )

            scheduler_output = scheduler.step(
                model_output=predicted_noise,
                timestep=current_timestep,
                sample=current_sample,
                generator=generator,
            )

            if not hasattr(
                scheduler_output,
                "prev_sample",
            ):
                raise TypeError(
                    "scheduler.step() must return "
                    "an object with prev_sample."
                )

            current_sample = (
                scheduler_output.prev_sample
            )

        # تبدیل خروجی از [-1,1] به [0,1]
        purified = (
            current_sample.clamp(
                -1.0,
                1.0,
            )
            + 1.0
        ) / 2.0

        purified = purified.clamp(
            0.0,
            1.0,
        )

        purified = purified.to(
            dtype=input_dtype
        )

    return purified


def purify_images(
    images: Tensor,
    unet: nn.Module,
    scheduler: Any,
    timestep: int = 50,
    batch_size: Optional[int] = None,
    generator: Optional[
        torch.Generator
    ] = None,
) -> Tensor:
    """
    Purifies clean or adversarial CIFAR-10 images
    using a pretrained DDPM.

    Parameters
    ----------
    images:
        Raw images with shape:

            (N, 3, 32, 32)

        Pixel values must be in [0, 1].

    unet:
        Pretrained diffusion UNet loaded by
        load_diffusion_components().

    scheduler:
        Matching DDPMScheduler loaded from the same
        Hugging Face repository.

    timestep:
        Forward-noise timestep.

        Project default:
            timestep = 50

    batch_size:
        Optional purification batch size.

        Smaller values reduce GPU memory usage.
        If None, the entire input batch is processed together.

    generator:
        Optional torch.Generator for reproducible
        forward and reverse noise.

        Its device type must match the UNet device.

    Returns
    -------
    Tensor:
        Purified images with:

        - the same shape as images
        - the same dtype as images
        - the same device as images
        - pixel values in [0, 1]

    Notes
    -----
    This function only purifies images.

    It does not:

    - run ResNet20
    - calculate accuracy
    - calculate robust accuracy
    - save figures
    - write metrics.csv
    """

    _validate_images(
        images
    )

    if batch_size is None:
        resolved_batch_size = (
            images.shape[0]
        )

    else:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size must be "
                "a positive integer or None."
            )

        resolved_batch_size = batch_size

    previous_mode = unet.training

    unet.eval()

    purified_batches: list[Tensor] = []

    try:
        for start_index in range(
            0,
            images.shape[0],
            resolved_batch_size,
        ):
            end_index = min(
                start_index
                + resolved_batch_size,
                images.shape[0],
            )

            purified_batch = _purify_batch(
                images=images[
                    start_index:end_index
                ],
                unet=unet,
                scheduler=scheduler,
                timestep=timestep,
                generator=generator,
            )

            purified_batches.append(
                purified_batch
            )

    finally:
        # حالت قبلی مدل diffusion بازیابی می‌شود.
        unet.train(
            previous_mode
        )

    purified_images = torch.cat(
        purified_batches,
        dim=0,
    ).detach()

    if purified_images.shape != images.shape:
        raise RuntimeError(
            "Purified output shape does not "
            "match input shape."
        )

    if purified_images.dtype != images.dtype:
        raise RuntimeError(
            "Purified output dtype does not "
            "match input dtype."
        )

    if purified_images.device != images.device:
        raise RuntimeError(
            "Purified output device does not "
            "match input device."
        )

    if not bool(
        torch.isfinite(
            purified_images
        ).all().item()
    ):
        raise RuntimeError(
            "Diffusion purification produced "
            "NaN or infinite values."
        )

    output_min = float(
        purified_images.min().item()
    )

    output_max = float(
        purified_images.max().item()
    )

    if output_min < 0.0 or output_max > 1.0:
        raise RuntimeError(
            "Diffusion purification produced "
            "values outside [0, 1]."
        )

    return purified_images