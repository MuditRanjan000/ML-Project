import torch
import torch.nn.functional as F

def compute_ece(logits: torch.Tensor, targets: torch.Tensor, temperature: float = 1.0, n_bins: int = 15) -> float:
    """
    Computes Expected Calibration Error (ECE).
    - 15 equal-width bins
    - confidence = max softmax probability
    - empty bins contribute zero
    """
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=1)
    confidences, predictions = torch.max(probabilities, dim=1)
    accuracies = predictions.eq(targets)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Enforce exact bin matching
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += prop_in_bin * torch.abs(avg_confidence_in_bin - accuracy_in_bin)

    return ece.item()
