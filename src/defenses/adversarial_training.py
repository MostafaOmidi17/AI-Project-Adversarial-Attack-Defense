import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Tuple

try:
    from src.attacks.pgd import pgd_attack
except ImportError:
    from attacks.pgd import pgd_attack


def train_adversarial_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    attack_steps: int = 10,
    random_start: bool = True,
) -> Tuple[float, float]:
    """
    Executes a single epoch of Adversarial Training using PGD.
    
    Args:
        model: The complete neural network model (including Normalize).
        dataloader: DataLoader for the training dataset.
        optimizer: The optimizer used to update model weights.
        criterion: The loss function (e.g., CrossEntropyLoss).
        device: CPU or GPU device.
        epsilon: PGD attack budget.
        alpha: PGD attack step size.
        attack_steps: Number of PGD steps (Contract Section 6: 10 steps for AT).
        random_start: Whether to use random initialization for PGD.
        
    Returns:
        Tuple containing average epoch loss and robust accuracy.
    """
    model.train()
    
    total_loss = 0.0
    correct_adv = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Generate adversarial examples for the current batch
        # Note: pgd_attack internally handles the model.eval() -> model.train() state
        adv_images = pgd_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=epsilon,
            alpha=alpha,
            steps=attack_steps,
            random_start=random_start,
            restarts=1
        )

        # Ensure gradients from previous iterations are cleared
        optimizer.zero_grad()

        # Forward pass using ADVERSARIAL images
        logits = model(adv_images)
        loss = criterion(logits, labels)

        # Backward pass to calculate gradients of the network parameters
        loss.backward()
        
        # Update network parameters
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct_adv += predictions.eq(labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = total_loss / total_samples
    epoch_robust_acc = correct_adv / max(total_samples, 1)

    return epoch_loss, epoch_robust_acc


def evaluate_robustness(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    attack_steps: int = 20,
) -> Tuple[float, float]:
    """
    Evaluates the model's robust accuracy on a validation/test set.
    Used during training to track the best model (Robust Overfitting check).
    
    Args:
        model: The complete neural network model.
        dataloader: DataLoader for the validation dataset.
        criterion: The loss function.
        device: CPU or GPU device.
        epsilon: PGD attack budget.
        alpha: PGD attack step size.
        attack_steps: Number of PGD steps (Contract Section 6: 20 steps for eval).
        
    Returns:
        Tuple containing average validation loss and robust accuracy.
    """
    was_training = model.training
    model.eval()
    
    total_loss = 0.0
    correct_adv = 0
    total_samples = 0

    try:
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Generate adversarial examples for evaluation (20 steps as per contract)
            adv_images = pgd_attack(
                model=model,
                images=images,
                labels=labels,
                epsilon=epsilon,
                alpha=alpha,
                steps=attack_steps,
                random_start=True,
                restarts=1
            )

            with torch.no_grad():
                logits = model(adv_images)
                loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            correct_adv += predictions.eq(labels).sum().item()
            total_samples += labels.size(0)
            
    finally:
        model.train(was_training)

    val_loss = total_loss / total_samples
    val_robust_acc = correct_adv / max(total_samples, 1)

    return val_loss, val_robust_acc