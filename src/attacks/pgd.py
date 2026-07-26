import torch
import torch.nn as nn
from torch import Tensor

def pgd_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 20,
    random_start: bool = True,
    restarts: int = 1
) -> Tensor:
    """
    Projected Gradient Descent (PGD) Attack.
    
    Args:
        model: The complete model (including normalization).
        images: Input images tensor in range [0, 1].
        labels: True labels of the images.
        epsilon: Maximum allowed perturbation (L_infinity norm).
        alpha: Step size for each iteration.
        steps: Number of attack iterations.
        random_start: Whether to add random initial noise.
        restarts: Number of random restarts to find the strongest attack.
        
    Returns:
        Adversarial images clipped to [0, 1] and bounded by epsilon.
    """
    # Return original images if epsilon is zero
    if epsilon == 0.0:
        return images.clone().detach()

    # Save model's training mode and switch to eval
    was_training = model.training
    model.eval()

    # Use 'mean' for gradient computation and 'none' to track the best loss across restarts
    criterion_mean = nn.CrossEntropyLoss(reduction='mean')
    criterion_none = nn.CrossEntropyLoss(reduction='none')

    best_adv_images = images.clone().detach()
    
    # Matrix to store the highest loss found for each image in the batch
    best_loss = torch.full((images.size(0),), -float('inf'), device=images.device)

    try:
        for _ in range(restarts):
            adv_images = images.clone().detach()

            if random_start:
                # Step 1: Add uniform random noise in range [-epsilon, epsilon]
                noise = torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
                adv_images = adv_images + noise
                
                # Clamp to ensure the image remains in valid pixel space [0, 1]
                adv_images = torch.clamp(adv_images, 0.0, 1.0)

            for step in range(steps):
                # Detach from previous graph and enable gradients
                adv_images = adv_images.detach()
                adv_images.requires_grad_(True)

                logits = model(adv_images)
                loss = criterion_mean(logits, labels)

                # Compute gradient without loss.backward()
                gradient = torch.autograd.grad(
                    outputs=loss,
                    inputs=adv_images,
                    only_inputs=True,
                )[0]

                # Take a small step in the direction of the gradient sign
                adv_images = adv_images + alpha * gradient.sign()

                # Projection (ensure perturbation stays within epsilon budget)
                perturbation = adv_images - images
                perturbation = torch.clamp(perturbation, -epsilon, epsilon)

                # Apply controlled perturbation and final clamp
                adv_images = images + perturbation
                adv_images = torch.clamp(adv_images, 0.0, 1.0)

            # Evaluate the success of the current restart
            with torch.no_grad():
                final_logits = model(adv_images)
                final_loss = criterion_none(final_logits, labels)

            # Store the adversarial image that caused the maximum loss
            mask = final_loss > best_loss
            best_loss[mask] = final_loss[mask]
            best_adv_images[mask] = adv_images[mask].detach()

    finally:
        # Restore original model state
        model.train(was_training)

    
    perturbation_check = (best_adv_images - images).abs().max()
    assert perturbation_check <= epsilon + 1e-6, "PGD perturbation exceeds epsilon budget!"

    return best_adv_images.detach()