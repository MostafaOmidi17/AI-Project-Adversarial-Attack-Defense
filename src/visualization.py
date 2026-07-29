from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from torch import Tensor


__all__ = [
    "CIFAR10_CLASS_NAMES",
    "plot_fgsm_epsilon_curve",
    "plot_robust_accuracy_heatmap",
    "plot_attack_examples",
    "plot_perturbation_norm_distribution",
    "plot_inference_time_comparison",
    "plot_robust_overfitting_curve",
    "plot_clean_robust_tradeoff",
    "plot_parameter_tradeoff",
]


CIFAR10_CLASS_NAMES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

DEFAULT_DEFENSE_ORDER: tuple[str, ...] = (
    "none",
    "pgd_at",
    "rand_smooth",
    "diff_purify",
)

DEFAULT_ATTACK_ORDER: tuple[str, ...] = (
    "clean",
    "fgsm",
    "pgd",
    "deepfool",
    "cw_l2",
)

DISPLAY_NAMES: dict[str, str] = {
    "none": "Base model",
    "pgd_at": "Adversarial training",
    "rand_smooth": "Randomized smoothing",
    "diff_purify": "Diffusion purification",
    "clean": "Clean",
    "fgsm": "FGSM",
    "pgd": "PGD",
    "deepfool": "DeepFool",
    "cw_l2": "C&W L2",
}


def _display_name(identifier: str) -> str:
    return DISPLAY_NAMES.get(
        identifier,
        identifier.replace("_", " ").title(),
    )


def _require_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    name: str,
) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError(f"{name} must not be empty.")

    missing = [column for column in columns if column not in dataframe]

    if missing:
        raise ValueError(f"{name} is missing columns: {missing}.")


def _numeric(
    series: pd.Series,
    name: str,
    allow_nan: bool = False,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    if not allow_nan and bool(values.isna().any()):
        raise ValueError(f"{name} contains missing or non-numeric values.")

    finite_values = values.dropna().to_numpy(dtype=float)

    if not np.isfinite(finite_values).all():
        raise ValueError(f"{name} contains non-finite values.")

    return values


def _accuracy(
    series: pd.Series,
    name: str,
    allow_nan: bool = False,
) -> pd.Series:
    values = _numeric(series, name=name, allow_nan=allow_nan)
    finite_values = values.dropna()

    if bool(((finite_values < 0.0) | (finite_values > 1.0)).any()):
        raise ValueError(f"{name} must contain values in [0, 1].")

    return values


def _prepare_save_path(
    save_path: Optional[str | Path],
) -> Optional[Path]:
    if save_path is None:
        return None

    path = Path(save_path)

    if path.suffix.lower() != ".png":
        raise ValueError("Figures must be saved as .png files.")

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _finish(
    figure: Figure,
    save_path: Optional[str | Path],
    show: bool,
    close: bool,
) -> None:
    figure.tight_layout()

    resolved_path = _prepare_save_path(save_path)

    if resolved_path is not None:
        figure.savefig(
            resolved_path,
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    if close:
        plt.close(figure)


def _epsilon_label(epsilon: float) -> str:
    pixel_units = epsilon * 255.0

    if math.isclose(
        pixel_units,
        round(pixel_units),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        return f"{int(round(pixel_units))}/255"

    return f"{epsilon:.4f}"


def _validate_attack_tensors(
    originals: Tensor,
    adversarials: Tensor,
    labels: Tensor,
    clean_logits: Tensor,
    adversarial_logits: Tensor,
    class_names: Sequence[str],
) -> None:
    if originals.ndim != 4 or originals.shape[0] == 0:
        raise ValueError("originals must have shape (N, C, H, W).")

    if originals.shape[1] != 3:
        raise ValueError("CIFAR-10 images must have 3 channels.")

    if adversarials.shape != originals.shape:
        raise ValueError(
            "adversarials must have the same shape as originals."
        )

    if originals.dtype != adversarials.dtype:
        raise TypeError(
            "originals and adversarials must have the same dtype."
        )

    if not torch.is_floating_point(originals):
        raise TypeError("Images must be floating-point tensors.")

    if labels.shape != (originals.shape[0],):
        raise ValueError("labels must have shape (N,).")

    if labels.dtype != torch.long:
        raise TypeError("labels must have dtype torch.long.")

    expected_shape = (
        originals.shape[0],
        len(class_names),
    )

    if tuple(clean_logits.shape) != expected_shape:
        raise ValueError(
            f"clean_logits must have shape {expected_shape}."
        )

    if tuple(adversarial_logits.shape) != expected_shape:
        raise ValueError(
            f"adversarial_logits must have shape {expected_shape}."
        )

    for name, tensor in (
        ("originals", originals),
        ("adversarials", adversarials),
        ("clean_logits", clean_logits),
        ("adversarial_logits", adversarial_logits),
    ):
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} contains NaN or infinite values.")

    for name, tensor in (
        ("originals", originals),
        ("adversarials", adversarials),
    ):
        minimum = float(tensor.detach().min().item())
        maximum = float(tensor.detach().max().item())

        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(f"{name} must be in [0, 1].")


def plot_fgsm_epsilon_curve(
    metrics_df: pd.DataFrame,
    defense_id: str = "none",
    value_column: str = "robust_accuracy",
    save_path: Optional[str | Path] = (
        "figures/fgsm_epsilon_curve.png"
    ),
    show: bool = False,
    close: bool = True,
) -> Figure:
    """
    Plot robust accuracy versus FGSM epsilon.

    Repeated runs with the same epsilon are averaged. Standard-deviation
    error bars are included when multiple seeds are available.
    """

    _require_columns(
        metrics_df,
        ("attack_id", "defense_id", "epsilon", value_column),
        "metrics_df",
    )

    data = metrics_df.loc[
        metrics_df["attack_id"].eq("fgsm")
        & metrics_df["defense_id"].eq(defense_id)
    ].copy()

    if data.empty:
        raise ValueError(
            f"No FGSM results found for defense_id={defense_id!r}."
        )

    data["epsilon"] = _numeric(data["epsilon"], "epsilon")
    data[value_column] = _accuracy(data[value_column], value_column)

    grouped = (
        data.groupby("epsilon")[value_column]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("epsilon")
    )

    grouped["std"] = grouped["std"].fillna(0.0)

    x = grouped["epsilon"].to_numpy(dtype=float)
    y = grouped["mean"].to_numpy(dtype=float) * 100.0
    error = grouped["std"].to_numpy(dtype=float) * 100.0

    figure, axis = plt.subplots(figsize=(7.2, 4.8))

    axis.errorbar(
        x,
        y,
        yerr=error,
        marker="o",
        capsize=4,
    )

    axis.set_title(
        f"FGSM epsilon sweep — {_display_name(defense_id)}"
    )
    axis.set_xlabel("Perturbation budget")
    axis.set_ylabel("Robust accuracy (%)")
    axis.set_xticks(x)
    axis.set_xticklabels([_epsilon_label(value) for value in x])
    axis.set_ylim(0.0, 100.0)
    axis.grid(True, alpha=0.3)

    _finish(figure, save_path, show, close)
    return figure


def plot_robust_accuracy_heatmap(
    metrics_df: pd.DataFrame,
    defense_order: Sequence[str] = DEFAULT_DEFENSE_ORDER,
    attack_order: Sequence[str] = DEFAULT_ATTACK_ORDER,
    save_path: Optional[str | Path] = (
        "figures/robust_accuracy_heatmap.png"
    ),
    show: bool = False,
    close: bool = True,
    *,
    fgsm_epsilon: float = 8 / 255,
) -> Figure:
    """
    Plot the required attack-by-defense evaluation matrix.

    Clean rows use clean_accuracy. Adversarial rows use robust_accuracy.

    The final project matrix uses FGSM at fgsm_epsilon, which defaults
    to the contract value 8/255. Repeated runs of the same selected
    configuration, such as multiple seeds, are averaged.
    """

    _require_columns(
        metrics_df,
        (
            "defense_id",
            "attack_id",
            "clean_accuracy",
            "robust_accuracy",
        ),
        "metrics_df",
    )

    data = metrics_df.copy()
    clean_accuracy = _accuracy(
        data["clean_accuracy"],
        "clean_accuracy",
        allow_nan=True,
    )
    robust_accuracy = _accuracy(
        data["robust_accuracy"],
        "robust_accuracy",
        allow_nan=True,
    )

    data["plot_accuracy"] = np.where(
        data["attack_id"].eq("clean"),
        clean_accuracy,
        robust_accuracy,
    )

    data = data.loc[
        data["defense_id"].isin(defense_order)
        & data["attack_id"].isin(attack_order)
    ]

    if (
        isinstance(fgsm_epsilon, bool)
        or not isinstance(fgsm_epsilon, (int, float))
        or not math.isfinite(float(fgsm_epsilon))
        or float(fgsm_epsilon) <= 0.0
    ):
        raise ValueError(
            "fgsm_epsilon must be a finite positive number."
        )

    fgsm_epsilon = float(fgsm_epsilon)
    fgsm_mask = data["attack_id"].eq("fgsm")

    if bool(fgsm_mask.any()):
        _require_columns(
            data,
            ("epsilon",),
            "metrics_df",
        )

        fgsm_values = _numeric(
            data.loc[fgsm_mask, "epsilon"],
            "FGSM epsilon",
        )

        fgsm_match_mask = pd.Series(
            False,
            index=data.index,
            dtype=bool,
        )

        fgsm_match_mask.loc[fgsm_mask] = np.isclose(
            fgsm_values.to_numpy(dtype=float),
            fgsm_epsilon,
            rtol=0.0,
            atol=1e-9,
        )

        if not bool(fgsm_match_mask.any()):
            raise ValueError(
                "No FGSM rows match the requested "
                f"fgsm_epsilon={_epsilon_label(fgsm_epsilon)}."
            )

        data = data.loc[
            ~fgsm_mask | fgsm_match_mask
        ]

    if data.empty:
        raise ValueError("No rows match the requested heatmap order.")

    matrix = (
        data.pivot_table(
            index="defense_id",
            columns="attack_id",
            values="plot_accuracy",
            aggfunc="mean",
        )
        .reindex(
            index=list(defense_order),
            columns=list(attack_order),
        )
        .to_numpy(dtype=float)
    )

    if not np.isfinite(matrix).any():
        raise ValueError("The heatmap has no finite accuracy values.")

    percentages = matrix * 100.0

    figure, axis = plt.subplots(
        figsize=(
            max(8.0, 1.4 * len(attack_order)),
            max(4.8, 0.9 * len(defense_order) + 1.6),
        )
    )

    image = axis.imshow(
        percentages,
        aspect="auto",
        vmin=0.0,
        vmax=100.0,
        cmap="viridis",
    )

    axis.set_title("Robust accuracy across attacks and defenses")
    axis.set_xlabel("Attack")
    axis.set_ylabel("Defense")
    axis.set_xticks(np.arange(len(attack_order)))
    axis.set_yticks(np.arange(len(defense_order)))
    axis.set_xticklabels(
        [_display_name(value) for value in attack_order]
    )
    axis.set_yticklabels(
        [_display_name(value) for value in defense_order]
    )

    for row in range(len(defense_order)):
        for column in range(len(attack_order)):
            value = percentages[row, column]
            label = "N/A" if not np.isfinite(value) else f"{value:.1f}%"

            if not np.isfinite(value):
                text_color = "black"
            else:
                text_color = "white" if value < 50.0 else "black"

            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontweight="semibold",
            )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Accuracy (%)")

    _finish(figure, save_path, show, close)
    return figure


def plot_attack_examples(
    originals: Tensor,
    adversarials: Tensor,
    labels: Tensor,
    clean_logits: Tensor,
    adversarial_logits: Tensor,
    attack_id: str,
    save_path: Optional[str | Path] = None,
    class_names: Sequence[str] = CIFAR10_CLASS_NAMES,
    max_examples: int = 2,
    amplification_factor: float = 10.0,
    strict_success_count: bool = True,
    show: bool = False,
    close: bool = True,
) -> tuple[Figure, list[int]]:
    """
    Plot original image, amplified perturbation, and adversarial image.

    A successful sample must be correctly classified before the attack and
    incorrectly classified afterward. The project requires two successful
    examples for each attack.
    """

    _validate_attack_tensors(
        originals,
        adversarials,
        labels,
        clean_logits,
        adversarial_logits,
        class_names,
    )

    if (
        isinstance(max_examples, bool)
        or not isinstance(max_examples, int)
        or max_examples <= 0
    ):
        raise ValueError("max_examples must be a positive integer.")

    if (
        isinstance(amplification_factor, bool)
        or not isinstance(amplification_factor, (int, float))
        or not math.isfinite(float(amplification_factor))
        or amplification_factor <= 0.0
    ):
        raise ValueError(
            "amplification_factor must be a finite positive number."
        )

    clean_probabilities = torch.softmax(
        clean_logits.detach(),
        dim=1,
    )
    adversarial_probabilities = torch.softmax(
        adversarial_logits.detach(),
        dim=1,
    )

    clean_confidence, clean_prediction = clean_probabilities.max(dim=1)
    adv_confidence, adv_prediction = adversarial_probabilities.max(dim=1)

    successful = (
        clean_prediction.eq(labels)
        & adv_prediction.ne(labels)
    )

    candidate_indices = torch.nonzero(
        successful,
        as_tuple=False,
    ).flatten()

    available = int(candidate_indices.numel())

    if available == 0:
        raise ValueError(
            f"No successful {attack_id} examples were provided."
        )

    if strict_success_count and available < max_examples:
        raise ValueError(
            f"Expected {max_examples} successful {attack_id} samples, "
            f"but only {available} were available."
        )

    selected = candidate_indices[: min(max_examples, available)]
    perturbation = adversarials - originals

    l2_norm = perturbation.flatten(1).norm(p=2, dim=1)
    linf_norm = perturbation.flatten(1).abs().max(dim=1).values

    figure, axes = plt.subplots(
        len(selected),
        3,
        figsize=(12.0, 3.8 * len(selected)),
        squeeze=False,
    )

    for row, tensor_index in enumerate(selected):
        index = int(tensor_index.item())

        original = (
            originals[index]
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )
        adversarial = (
            adversarials[index]
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )
        noise = (
            perturbation[index]
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        displayed_noise = np.clip(
            0.5 + amplification_factor * noise,
            0.0,
            1.0,
        )

        true_index = int(labels[index].item())
        clean_index = int(clean_prediction[index].item())
        adv_index = int(adv_prediction[index].item())

        axes[row, 0].imshow(original)
        axes[row, 0].set_title(
            "Original\n"
            f"True: {class_names[true_index]} | "
            f"Pred: {class_names[clean_index]}\n"
            f"Confidence: {clean_confidence[index].item() * 100:.1f}%"
        )

        axes[row, 1].imshow(displayed_noise)
        axes[row, 1].set_title(
            f"Perturbation ×{amplification_factor:g}\n"
            f"L2: {l2_norm[index].item():.4f} | "
            f"Linf: {linf_norm[index].item():.5f}"
        )

        axes[row, 2].imshow(adversarial)
        axes[row, 2].set_title(
            "Adversarial\n"
            f"Pred: {class_names[adv_index]}\n"
            f"Confidence: {adv_confidence[index].item() * 100:.1f}%"
        )

        for column in range(3):
            axes[row, column].axis("off")

    figure.suptitle(
        f"Successful {_display_name(attack_id)} examples",
        fontsize=14,
    )

    if save_path is None:
        save_path = f"figures/attack_examples_{attack_id}.png"

    _finish(figure, save_path, show, close)

    return figure, [int(index.item()) for index in selected]


def plot_perturbation_norm_distribution(
    norms: Tensor | Sequence[float] | np.ndarray,
    attack_id: str,
    norm_name: str = "l2",
    successful_mask: Optional[
        Tensor | Sequence[bool] | np.ndarray
    ] = None,
    bins: int = 30,
    save_path: Optional[str | Path] = None,
    show: bool = False,
    close: bool = True,
) -> Figure:
    """
    Plot perturbation norms, optionally restricted to successful attacks.
    """

    if torch.is_tensor(norms):
        values = norms.detach().cpu().flatten().numpy()
    else:
        values = np.asarray(norms, dtype=float).reshape(-1)

    if values.size == 0:
        raise ValueError("norms must not be empty.")

    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("norms must be finite and non-negative.")

    if successful_mask is not None:
        if torch.is_tensor(successful_mask):
            mask = (
                successful_mask.detach()
                .cpu()
                .flatten()
                .numpy()
                .astype(bool)
            )
        else:
            mask = np.asarray(successful_mask, dtype=bool).reshape(-1)

        if mask.shape != values.shape:
            raise ValueError(
                "successful_mask must have the same length as norms."
            )

        values = values[mask]

        if values.size == 0:
            raise ValueError("No successful samples remain after masking.")

    if (
        isinstance(bins, bool)
        or not isinstance(bins, int)
        or bins <= 0
    ):
        raise ValueError("bins must be a positive integer.")

    mean_value = float(np.mean(values))
    median_value = float(np.median(values))

    figure, axis = plt.subplots(figsize=(7.2, 4.8))

    axis.hist(values, bins=bins, edgecolor="black", alpha=0.8)
    axis.axvline(
        mean_value,
        linestyle="--",
        label=f"Mean: {mean_value:.4f}",
    )
    axis.axvline(
        median_value,
        linestyle=":",
        label=f"Median: {median_value:.4f}",
    )

    axis.set_title(
        f"{norm_name.upper()} perturbation distribution — "
        f"{_display_name(attack_id)}"
    )
    axis.set_xlabel(f"{norm_name.upper()} norm")
    axis.set_ylabel("Number of samples")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()

    if save_path is None:
        save_path = (
            f"figures/{attack_id}_{norm_name.lower()}_distribution.png"
        )

    _finish(figure, save_path, show, close)
    return figure


def plot_inference_time_comparison(
    metrics_df: pd.DataFrame,
    time_column: str = "total_time_seconds",
    save_path: Optional[str | Path] = (
        "figures/inference_time_comparison.png"
    ),
    show: bool = False,
    close: bool = True,
) -> Figure:
    """
    Compare mean inference time per image across defenses.
    """

    _require_columns(
        metrics_df,
        ("defense_id", "num_samples", time_column),
        "metrics_df",
    )

    data = metrics_df.copy()
    data["num_samples"] = _numeric(
        data["num_samples"],
        "num_samples",
    )
    data[time_column] = _numeric(
        data[time_column],
        time_column,
    )

    if bool((data["num_samples"] <= 0).any()):
        raise ValueError("num_samples must be positive.")

    if bool((data[time_column] < 0.0).any()):
        raise ValueError(f"{time_column} must be non-negative.")

    data["milliseconds_per_image"] = (
        data[time_column]
        / data["num_samples"]
        * 1000.0
    )

    grouped = (
        data.groupby("defense_id", as_index=False)[
            "milliseconds_per_image"
        ]
        .mean()
        .sort_values("milliseconds_per_image")
    )

    labels = [
        _display_name(value)
        for value in grouped["defense_id"]
    ]
    values = grouped["milliseconds_per_image"].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    axis.barh(labels, values)

    for index, value in enumerate(values):
        axis.text(value, index, f" {value:.2f} ms", va="center")

    axis.set_title("Average inference time per image")
    axis.set_xlabel("Milliseconds per image")
    axis.set_ylabel("Defense")
    axis.grid(True, axis="x", alpha=0.3)

    _finish(figure, save_path, show, close)
    return figure


def _history_array(
    history: Mapping[str, Sequence[float]],
    key: str,
    expected_length: Optional[int] = None,
) -> np.ndarray:
    if key not in history:
        raise ValueError(f"history is missing {key!r}.")

    values = np.asarray(history[key], dtype=float).reshape(-1)

    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"history[{key!r}] is empty or non-finite.")

    if expected_length is not None and values.size != expected_length:
        raise ValueError(
            f"history[{key!r}] must contain {expected_length} values."
        )

    if np.min(values) < 0.0 or np.max(values) > 1.0:
        raise ValueError(f"history[{key!r}] must be in [0, 1].")

    return values


def plot_robust_overfitting_curve(
    history: Mapping[str, Sequence[float]],
    save_path: Optional[str | Path] = (
        "figures/robust_overfitting_curve.png"
    ),
    show: bool = False,
    close: bool = True,
) -> Figure:
    """
    Diagnose robust overfitting from train and validation accuracies.

    Required keys:
        train_robust_accuracy
        val_robust_accuracy

    Optional keys:
        train_clean_accuracy
        val_clean_accuracy
    """

    if not isinstance(history, Mapping):
        raise TypeError("history must be a mapping.")

    train_robust = _history_array(
        history,
        "train_robust_accuracy",
    )
    val_robust = _history_array(
        history,
        "val_robust_accuracy",
        expected_length=len(train_robust),
    )

    epochs = np.arange(1, len(train_robust) + 1)

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(
        epochs,
        train_robust * 100.0,
        marker="o",
        label="Train robust accuracy",
    )
    axis.plot(
        epochs,
        val_robust * 100.0,
        marker="o",
        label="Validation robust accuracy",
    )

    optional_series = (
        ("train_clean_accuracy", "Train clean accuracy"),
        ("val_clean_accuracy", "Validation clean accuracy"),
    )

    for key, label in optional_series:
        if key in history:
            values = _history_array(
                history,
                key,
                expected_length=len(train_robust),
            )
            axis.plot(
                epochs,
                values * 100.0,
                linestyle="--",
                label=label,
            )

    best_epoch = int(np.argmax(val_robust)) + 1

    axis.axvline(
        best_epoch,
        linestyle=":",
        label=f"Best validation robust epoch: {best_epoch}",
    )

    axis.set_title("Robust-overfitting diagnostic")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(0.0, 100.0)
    axis.grid(True, alpha=0.3)
    axis.legend()

    _finish(figure, save_path, show, close)
    return figure


def plot_clean_robust_tradeoff(
    metrics_df: pd.DataFrame,
    attack_id: str = "pgd",
    save_path: Optional[str | Path] = (
        "figures/clean_robust_tradeoff.png"
    ),
    show: bool = False,
    close: bool = True,
) -> Figure:
    """
    Plot clean accuracy against robust accuracy for each defense.
    """

    _require_columns(
        metrics_df,
        (
            "defense_id",
            "attack_id",
            "clean_accuracy",
            "robust_accuracy",
        ),
        "metrics_df",
    )

    data = metrics_df.loc[
        metrics_df["attack_id"].eq(attack_id)
    ].copy()

    if data.empty:
        raise ValueError(
            f"No rows found for attack_id={attack_id!r}."
        )

    data["clean_accuracy"] = _accuracy(
        data["clean_accuracy"],
        "clean_accuracy",
    )
    data["robust_accuracy"] = _accuracy(
        data["robust_accuracy"],
        "robust_accuracy",
    )

    grouped = (
        data.groupby("defense_id", as_index=False)[
            ["clean_accuracy", "robust_accuracy"]
        ]
        .mean()
    )

    figure, axis = plt.subplots(figsize=(6.8, 5.4))

    axis.scatter(
        grouped["clean_accuracy"] * 100.0,
        grouped["robust_accuracy"] * 100.0,
        s=80,
    )

    for _, row in grouped.iterrows():
        axis.annotate(
            _display_name(str(row["defense_id"])),
            (
                float(row["clean_accuracy"]) * 100.0,
                float(row["robust_accuracy"]) * 100.0,
            ),
            xytext=(6, 6),
            textcoords="offset points",
        )

    axis.set_title(
        f"Clean–robust trade-off under {_display_name(attack_id)}"
    )
    axis.set_xlabel("Clean accuracy (%)")
    axis.set_ylabel("Robust accuracy (%)")
    axis.set_xlim(0.0, 100.0)
    axis.set_ylim(0.0, 100.0)
    axis.grid(True, alpha=0.3)

    _finish(figure, save_path, show, close)
    return figure


def plot_parameter_tradeoff(
    results_df: pd.DataFrame,
    parameter_column: str,
    clean_column: str = "clean_accuracy",
    robust_column: str = "robust_accuracy",
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    show: bool = False,
    close: bool = True,
) -> Figure:
    """
    Plot clean/robust accuracy against a defense parameter.

    Examples:
        sigma
        num_samples
        timestep
    """

    _require_columns(
        results_df,
        (parameter_column, clean_column, robust_column),
        "results_df",
    )

    data = results_df.copy()
    data[parameter_column] = _numeric(
        data[parameter_column],
        parameter_column,
    )
    data[clean_column] = _accuracy(
        data[clean_column],
        clean_column,
    )
    data[robust_column] = _accuracy(
        data[robust_column],
        robust_column,
    )

    grouped = (
        data.groupby(parameter_column, as_index=False)[
            [clean_column, robust_column]
        ]
        .mean()
        .sort_values(parameter_column)
    )

    x = grouped[parameter_column].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    axis.plot(
        x,
        grouped[clean_column] * 100.0,
        marker="o",
        label="Clean accuracy",
    )
    axis.plot(
        x,
        grouped[robust_column] * 100.0,
        marker="o",
        label="Robust accuracy",
    )

    axis.set_title(
        title or f"Trade-off over {parameter_column}"
    )
    axis.set_xlabel(
        x_label
        or parameter_column.replace("_", " ").title()
    )
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(0.0, 100.0)
    axis.grid(True, alpha=0.3)
    axis.legend()

    _finish(figure, save_path, show, close)
    return figure