# Evaluation Protocol

This document outlines the strict evaluation protocol for assessing model robustness and calibration.

## 1. CIFAR-100-C Corruptions
The evaluation suite must evaluate models across the following 15 standard corruptions defined by Hendrycks & Dietterich:
1. **Noise:** Gaussian, Shot, Impulse
2. **Blur:** Defocus, Glass, Motion, Zoom
3. **Weather:** Snow, Frost, Fog, Brightness
4. **Digital:** Contrast, Elastic, Pixelate, JPEG

## 2. Severity Levels
Each corruption above contains 5 severity levels (1 through 5). 
A complete CIFAR-100-C evaluation involves:
- 15 corruptions * 5 severities * 10,000 images = **750,000 evaluation inferences** per model.

## 3. ECE Binning Rules
When computing Expected Calibration Error (ECE):
- The confidence space `[0.0, 1.0]` is divided into exactly **15 uniform bins**.
- Confidence is defined strictly as the maximum softmax probability of the model's output vector.
- Empty bins (bins containing zero samples) are excluded from the ECE calculation.

## 4. Evaluation Table Schema
The final evaluator script must produce a tabular summary containing the following exact columns:
- `Model Name`
- `Clean Top-1 Acc`
- `Mean Corrupted Error`
- `Absolute Error Increase (AEI)`
- `Raw ECE`
- `Temp-Scaled ECE`

## 5. Result Storage Format
- Evaluator outputs must be saved strictly to the `results/` directory.
- Results must be serialized as `JSON` (for programmatic parsing) and `CSV` (for immediate human-readable tabular viewing).
- Evaluation metadata (timestamp, model path, config path, and the seed evaluated) must be tracked inside the JSON object to ensure unbroken provenance.
