# METRIC SPECIFICATIONS

This document precisely defines all evaluation metrics for the project to ensure unambiguous, reproducible results.

## 1. Clean Accuracy
*   **Formula:** `Count(Correct Predictions) / Total Samples`
*   **Input Split:** CIFAR-100 Test Set (10,000 images, clean).
*   **Aggregation:** Macro-averaged over all classes.
*   **Direction of Improvement:** Higher is better.
*   **Computation:** Computed per-seed, then aggregated (mean/std/CI) across final seeds.

## 2. Standard mCE (Mean Corruption Error)
*   **Formula:** `sum_{c=1}^{15} ( sum_{s=1}^{5} E_{c,s}^{model} ) / ( sum_{s=1}^{5} E_{c,s}^{baseline} ) / 15` where `E` is top-1 error. The baseline is AlexNet as defined in Hendrycks & Dietterich (2019).
*   **Input Split:** CIFAR-100-C.
*   **Aggregation:** Averaged across 15 corruptions and 5 severities.
*   **Direction of Improvement:** Lower is better.
*   **Computation:** Computed per-seed.

## 3. Absolute Error Increase (AEI)
*   **Formula:** `Corrupted Error - Clean Error`
*   **Input Split:** CIFAR-100-C (for Corrupted Error) and CIFAR-100 Test (for Clean Error).
*   **Aggregation:** Averaged across all corruptions and severities.
*   **Direction of Improvement:** Lower is better.
*   **Computation:** Computed per-seed to explicitly decouple baseline capability from degradation.

## 4. ECE (Expected Calibration Error)
*   **Formula:** `sum_{m=1}^{15} (|B_m| / N) * |acc(B_m) - conf(B_m)|`
*   **Input Split:** CIFAR-100-C (evaluated post-temperature scaling). Temperature scaling $T$ is fit ONLY on the 5k CIFAR-100 Clean Validation set. NO test-set tuning.
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
