import torch
import torch.nn as nn
from torch import Tensor

def fgsm_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float,
    criterion: nn.Module = nn.CrossEntropyLoss()
) -> Tensor:
    """
    Fast Gradient Sign Method (FGSM) Attack.
    """
    if epsilon == 0.0:
        return images.clone().detach()

    was_training = model.training
    model.eval()

    try:
        images = images.clone().detach().to(images.device)
        images.requires_grad_(True)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        gradient = torch.autograd.grad(
            outputs=loss,
            inputs=images,
            only_inputs=True,
        )[0]
        
        adv_images = images + epsilon * gradient.sign()
        
        adv_images = torch.clamp(adv_images, 0.0, 1.0)
        
    finally:
        model.train(was_training)
        
    return adv_images.detach()