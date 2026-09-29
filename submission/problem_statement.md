# Architecture or Training Recipe? 
**A Controlled Study of Corruption Robustness and Calibration in CNNs and Vision Transformers**

## 1. Background
Convolutional Neural Networks (CNNs) have long been the standard architecture for image classification. Recently, Vision Transformers (ViTs) have emerged as a powerful alternative, often demonstrating superior performance on large-scale datasets. Concurrently, evaluating model reliability under distribution shift—specifically corruption robustness—has become a critical area of research. Empirical studies often observe that ViTs exhibit greater robustness to common corruptions than CNNs. However, the exact source of this advantage remains heavily debated, as ViTs are typically trained with vastly different, highly regularized modern training recipes compared to the conventional supervised recipes historically used for CNNs.

## 2. Problem Statement
Apparent robustness differences between CNNs and Vision Transformers may be profoundly affected by differences in training recipes, optimization strategies, augmentations, model configurations, and experimental setups. When a ViT trained with a modern recipe (e.g., Mixup, CutMix, RandAugment, AdamW) is compared against a CNN trained with a conventional recipe (e.g., standard crop/flip, SGD), it is impossible to determine whether observed robustness gains stem from the self-attention mechanism's inductive bias or merely the superior regularization. A naïve comparison can therefore be misleading, incorrectly attributing robustness to architecture when it may be a product of the training framework.

## 3. Research Motivation
Understanding whether robustness differences come from architecture, training recipe, or their interaction is crucial. If robustness is primarily a product of the training recipe, then the field can improve CNN reliability simply by updating training protocols, rather than requiring a complete architectural paradigm shift to ViTs. Conversely, if the architectural inductive bias is the primary driver, it justifies the continued structural shift toward attention-based models in safety-critical applications.

## 4. Research Gap
We investigate whether the robustness gap between CNNs and ViTs persists when both architectures are strictly controlled for training recipe and evaluated at a standard scale. While prior work has highlighted the importance of training frameworks (e.g., Bai et al., 2021), the literature lacks a comprehensive, fully crossed factorial study that (a) evaluates standard entry-level CNN and ViT architectures without artificially matching capacity or using early stopping, (b) treats architecture and recipe as perfectly controlled factors, and (c) jointly analyzes both corruption robustness (accuracy degradation) and calibration (Expected Calibration Error) under shift to determine if these properties dissociate.

## 5. Primary Research Question
"Under controlled training conditions, how do CNNs and Vision Transformers differ in corruption robustness, and to what extent do these differences change when the same training recipe is applied across architectures?"

## 6. Secondary Research Question
"Do architecture- and recipe-related differences in corruption robustness correspond to differences in calibration under the same distribution shifts?"

## 7. Hypotheses
*   **H1 (Robustness):** A shared modern training recipe significantly alters the comparative corruption robustness of CNNs and ViTs. The mean Corruption Error (mCE) gap between the architectures will shrink significantly when both are trained with the same modern recipe, demonstrating that training protocol explains a substantial portion of the robustness gap.
*   **H2 (Calibration):** Architecture and recipe-induced differences in accuracy do not perfectly couple with calibration. The architectural effect on Expected Calibration Error (ECE) at high corruption severity will not be completely eliminated by recipe matching.

## 8. Proposed Methodology
To test these hypotheses, we propose an 8-week empirical study utilizing a 2x2x3 factorial design:
*   **Architectures:** ResNet-18 (standard CNN baseline) and a CIFAR-adapted ViT-Tiny (standard ViT baseline).
*   **Recipes:** 
    *   *Recipe A:* Conventional supervised training (SGD, MultiStep LR, standard crop/flip).
    *   *Recipe B:* Composite modern regularized training (AdamW, Cosine LR, RandAugment, Mixup, CutMix, Label Smoothing, Random Erasing, Stochastic Depth).
*   **Dataset:** CIFAR-100 (45k train, 5k clean validation for temperature scaling, 10k test).
*   **Evaluation:** CIFAR-100-C (15 corruptions, 5 severities).
*   **Seeds:** 3 fixed random seeds to estimate run-to-run variability.
*   **Metrics:** Clean Accuracy, Standard mCE, Absolute Error Increase (Corrupted Error - Clean Error), and 15-bin ECE.
*   **Analysis:** Cautious factorial analysis reporting effect sizes, uncertainty, and (if assumptions hold) a 2x2 Factorial ANOVA, alongside failure-mode analysis.

## 9. Fairness Protocol
Our methodology ensures a fair comparison through strict controls:
*   Fixed architectures and fixed training budgets (200 epochs) for all runs.
*   The exact same dataset and exact same composite recipe are applied across both architectures.
*   We rely on natural clean performance. We do not use artificial early stopping, nor do we alter model capacities (which would destroy the standard inductive biases).
*   No test-set tuning. Temperature scaling is fit strictly on a held-out clean validation split.

## 10. Expected Contribution
This project provides a controlled empirical analysis of architecture, training recipe, interaction effects, robustness, and calibration. We do not claim to produce a new state-of-the-art architecture. Instead, we contribute a rigorous quantification of *how* these factors govern model reliability.

## 11. Expected Outcome
The study is designed to determine whether observed robustness differences between CNNs and ViTs persist after strictly controlling the training conditions, ultimately clarifying the attribution of robustness in modern computer vision.
