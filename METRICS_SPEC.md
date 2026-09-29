# METRIC SPECIFICATIONS

This document precisely defines all evaluation metrics for the project to ensure unambiguous, reproducible results.

## 1. Clean Top-1 Accuracy
*   **Formula:** `Count(Correct Predictions) / Total Samples`
*   **Input Split:** CIFAR-100 Test Set (10,000 images, clean).
*   **Aggregation:** None (Simple top-1 accuracy).
*   **Direction of Improvement:** Higher is better.
*   **Computation:** Computed per-seed, then aggregated (mean/std/CI) across final seeds.

## 2. Mean Corrupted Error
*   **Formula:** Average of top-1 errors across all 15 corruptions and 5 severities.
*   **Input Split:** CIFAR-100-C.
*   **Aggregation:** Averaged across 15 corruptions and 5 severities.
*   **Direction of Improvement:** Lower is better.
*   **Computation:** Computed per-seed.
*   *Note: Do not call this unnormalized mean corrupted error "mCE". Reference-normalized mCE requires a CIFAR-100-C reference error table and should remain optional until a valid reference is selected.*

## 3. Absolute Error Increase (AEI)
*   **Formula:** `Corrupted Error - Clean Error`
*   **Input Split:** CIFAR-100-C (for Corrupted Error) and CIFAR-100 Test (for Clean Error).
*   **Aggregation:** Averaged across all corruptions and severities.
*   **Direction of Improvement:** Lower is better.
*   **Computation:** Computed per-seed to explicitly decouple baseline capability from degradation.

## 4. Calibration (ECE)
*   **Metrics:** Raw ECE and Temperature-scaled ECE are strictly frozen.
*   **Formula (Raw):** `sum_{m=1}^{15} (|B_m| / N) * |acc(B_m) - conf(B_m)|`
*   **Temperature Scaling:** 
    - Fitted ONLY on the 5,000-image clean validation set. 
    - NEVER fitted on CIFAR-100-C.
    - Does not affect accuracy metrics.
*   **Input Split:** CIFAR-100-C.
*   **Aggregation:** 15 uniform confidence bins [0, 1]. Confidence is the maximum softmax probability. Empty bins are excluded.
*   **Direction of Improvement:** Lower is better (0 is perfectly calibrated).
*   **Computation:** Computed per-seed.

## 5. Accuracy by Corruption
*   **Formula:** Standard top-1 accuracy calculated independently for each of the 15 corruption types.
*   **Input Split:** CIFAR-100-C.
*   **Aggregation:** Averaged across the 5 severities for a specific corruption type.
*   **Direction of Improvement:** Higher is better.
*   **Computation:** Computed per-seed.

## 6. Accuracy by Severity
*   **Formula:** Standard top-1 accuracy calculated independently for each of the 5 severity levels.
*   **Input Split:** CIFAR-100-C.
*   **Aggregation:** Averaged across the 15 corruption types for a specific severity level.
*   **Direction of Improvement:** Higher is better.
*   **Computation:** Computed per-seed.

## 7. Error by Corruption
*   **Formula:** `1.0 - Accuracy by Corruption`
*   **Input Split:** CIFAR-100-C.
*   **Aggregation:** Averaged across the 5 severities for a specific corruption type.
*   **Direction of Improvement:** Lower is better.
*   **Computation:** Computed per-seed.
