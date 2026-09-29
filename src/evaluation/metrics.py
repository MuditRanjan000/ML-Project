import torch

def calculate_top1_metrics(correct_predictions: int, total_samples: int) -> dict:
    """
    Computes standard top-1 metrics based strictly on raw counts.
    No macro-averaging across classes.
    """
    accuracy = correct_predictions / max(total_samples, 1)
    error = 1.0 - accuracy
    return {
        "accuracy": accuracy,
        "error": error
    }

def mean_corrupted_error(corruption_errors: list) -> float:
    """
    Computes unnormalized Mean Corrupted Error.
    Average of top-1 errors across all 15 corruptions and 5 severities.
    """
    if not corruption_errors:
        return 0.0
    return sum(corruption_errors) / len(corruption_errors)

def absolute_error_increase(corrupted_error: float, clean_error: float) -> float:
    """
    Computes AEI (Absolute Error Increase).
    """
    return corrupted_error - clean_error
