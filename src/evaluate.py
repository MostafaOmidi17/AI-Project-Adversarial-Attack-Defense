import os
import time
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Import Person 1 modules (Always available)
from src.models import build_normalized_resnet20
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack

# Import Person 2 modules (Handled with try-except for parallel development)
try:
    from src.attacks.deepfool import deepfool_attack
except ImportError:
    deepfool_attack = None

try:
    from src.attacks.cw_l2 import cw_l2_attack
except ImportError:
    cw_l2_attack = None


def run_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_id: str,
    defense_id: str,
    attack_id: str,
    split: str,
    seed: int,
    checkpoint_name: str,
    attack_fn=None,
    attack_params: dict = None,
):
    """
    Evaluates a model against a specific attack and logs the results to metrics.csv
    exactly according to Contract Sections 9, 10, and 17.
    """
    if attack_params is None:
        attack_params = {}

    model.eval()
    
    total_samples = 0
    clean_correct = 0
    adv_correct = 0
    successful_attacks = 0
    
    total_l2 = 0.0
    total_l2_successful = 0.0
    total_linf = 0.0
    total_linf_successful = 0.0
    
    attack_time_total = 0.0
    defense_time_total = 0.0 # Will be populated when evaluating Randomized Smoothing / DiffPure

    # Contract Section 25: Batch processing to manage GPU memory
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        
        # 1. Clean Pass (No attack)
        with torch.no_grad():
            clean_logits = model(images)
            clean_preds = clean_logits.argmax(dim=1)
            
        clean_correct_mask = clean_preds.eq(labels)
        clean_correct += clean_correct_mask.sum().item()

        # 2. Attack Pass
        if attack_id == "clean" or attack_fn is None:
            adv_images = images.clone()
            adv_preds = clean_preds
            attack_time = 0.0
        else:
            # Measure attack time
            start_time = time.time()
            adv_images = attack_fn(model, images, labels, **attack_params)
            attack_time_total += (time.time() - start_time)
            
            with torch.no_grad():
                adv_logits = model(adv_images)
                adv_preds = adv_logits.argmax(dim=1)

        # 3. Calculate Metrics (Contract Sections 9 & 10)
        adv_correct_mask = adv_preds.eq(labels)
        adv_correct += adv_correct_mask.sum().item()
        
        # Attack is successful ONLY if the clean model was correct AND adversarial is wrong
        success_mask = clean_correct_mask & (~adv_correct_mask)
        successful_attacks += success_mask.sum().item()

        # Calculate Perturbation Norms (Contract Section 10)
        perturbation = adv_images - images
        l2_norms = perturbation.flatten(1).norm(p=2, dim=1)
        linf_norms = perturbation.flatten(1).abs().max(dim=1).values
        
        total_l2 += l2_norms.sum().item()
        total_linf += linf_norms.sum().item()
        
        # Norms for successful attacks only
        if success_mask.any():
            total_l2_successful += l2_norms[success_mask].sum().item()
            total_linf_successful += linf_norms[success_mask].sum().item()

        total_samples += batch_size

    # Aggregate Metrics (Contract Section 17)
    clean_acc = clean_correct / total_samples
    robust_acc = adv_correct / total_samples
    asr = successful_attacks / max(clean_correct, 1)
    
    mean_l2 = total_l2 / total_samples
    mean_linf = total_linf / total_samples
    
    mean_l2_succ = (total_l2_successful / successful_attacks) if successful_attacks > 0 else float('NaN')
    mean_linf_succ = (total_linf_successful / successful_attacks) if successful_attacks > 0 else float('NaN')

    total_time = attack_time_total + defense_time_total

    # Construct run_id (Contract Section 16)
    param_str = ""
    if attack_id == "fgsm":
        param_str = f"__eps{int(attack_params.get('epsilon', 0) * 255)}"
    elif attack_id == "pgd":
        param_str = f"__eps{int(attack_params.get('epsilon', 0) * 255)}__steps{attack_params.get('steps', 0)}"
    run_id = f"{defense_id}__{attack_id}{param_str}__seed{seed}"

    # Prepare row for CSV
    row = {
        "run_id": run_id,
        "model_id": model_id,
        "defense_id": defense_id,
        "attack_id": attack_id,
        "split": split,
        "num_samples": total_samples,
        "seed": seed,
        "epsilon": attack_params.get("epsilon", float('NaN')),
        "alpha": attack_params.get("alpha", float('NaN')),
        "attack_steps": attack_params.get("steps", float('NaN')),
        "clean_accuracy": clean_acc,
        "robust_accuracy": robust_acc,
        "attack_success_rate": asr,
        "mean_l2": mean_l2,
        "mean_l2_successful": mean_l2_succ,
        "mean_linf": mean_linf,
        "mean_linf_successful": mean_linf_succ,
        "attack_time_seconds": attack_time_total,
        "defense_time_seconds": defense_time_total,
        "total_time_seconds": total_time,
        "checkpoint_name": checkpoint_name
    }

    # Save to CSV (Contract Section 17)
    csv_path = os.path.join("results", "metrics.csv")
    os.makedirs("results", exist_ok=True)
    
    df = pd.DataFrame([row])
    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)
        
    print(f"Evaluated {run_id} | Clean Acc: {clean_acc:.4f} | Robust Acc: {robust_acc:.4f} | ASR: {asr:.4f}")


def main():
    """
    Main evaluation routine for Person 1's tasks.
    Evaluates the base model and PGD-AT model against Clean, FGSM, and PGD.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Evaluation on {device}...")
    
    SEED = 42
    torch.manual_seed(SEED)

    # Load test dataset
    transform = transforms.ToTensor()
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # =========================================================================
    # Phase 1: Evaluate Base Model (none)
    # =========================================================================
    checkpoint_clean = os.path.join("checkpoints", "resnet20_clean_best.pt")
    if os.path.exists(checkpoint_clean):
        print("\n--- Evaluating Base Model ---")
        model_clean = build_normalized_resnet20(checkpoint_path=checkpoint_clean, eval_mode=True, device=device)
        
        # 1. Clean evaluation
        run_evaluation(model_clean, test_loader, device, "resnet20", "none", "clean", "test", SEED, "resnet20_clean_best.pt")
        
        # 2. FGSM evaluation
        run_evaluation(model_clean, test_loader, device, "resnet20", "none", "fgsm", "test", SEED, "resnet20_clean_best.pt", 
                       attack_fn=fgsm_attack, attack_params={"epsilon": 8/255})
        
        # 3. PGD evaluation
        run_evaluation(model_clean, test_loader, device, "resnet20", "none", "pgd", "test", SEED, "resnet20_clean_best.pt", 
                       attack_fn=pgd_attack, attack_params={"epsilon": 8/255, "alpha": 2/255, "steps": 20})
    else:
        print(f"Warning: Clean checkpoint not found at {checkpoint_clean}")

    # =========================================================================
    # Phase 2: Evaluate Adversarially Trained Model (pgd_at)
    # =========================================================================
    checkpoint_at = os.path.join("checkpoints", "resnet20_pgd_at_eps8_best.pt")
    if os.path.exists(checkpoint_at):
        print("\n--- Evaluating PGD-AT Model ---")
        model_at = build_normalized_resnet20(checkpoint_path=checkpoint_at, eval_mode=True, device=device)
        
        # 1. Clean evaluation
        run_evaluation(model_at, test_loader, device, "resnet20", "pgd_at", "clean", "test", SEED, "resnet20_pgd_at_eps8_best.pt")
        
        # 2. FGSM evaluation
        run_evaluation(model_at, test_loader, device, "resnet20", "pgd_at", "fgsm", "test", SEED, "resnet20_pgd_at_eps8_best.pt", 
                       attack_fn=fgsm_attack, attack_params={"epsilon": 8/255})
        
        # 3. PGD evaluation
        run_evaluation(model_at, test_loader, device, "resnet20", "pgd_at", "pgd", "test", SEED, "resnet20_pgd_at_eps8_best.pt", 
                       attack_fn=pgd_attack, attack_params={"epsilon": 8/255, "alpha": 2/255, "steps": 20})
    else:
        print(f"Warning: PGD-AT checkpoint not found at {checkpoint_at}")

if __name__ == "__main__":
    main()