# PROJECT SPECIFICATION

**PROJECT TITLE:** Architecture or Training Recipe? A Controlled Study of Corruption Robustness and Calibration in CNNs and Vision Transformers

**PRIMARY RESEARCH QUESTION:** 
"Under controlled training conditions, how do CNNs and Vision Transformers differ in corruption robustness, and to what extent do these differences change when the same training recipe is applied across architectures?"

**SECONDARY RESEARCH QUESTION:** 
"Do architecture- and recipe-related differences in corruption robustness correspond to differences in calibration under the same distribution shifts?"

**MAIN CONTRIBUTION:** 
"We provide a controlled empirical comparison that quantifies the effects of architecture, training recipe, and their interaction on corruption robustness and calibration."

**CNN ARCHITECTURE:** 
ResNet-18 (CIFAR 32x32 variant). Initial 3x3 conv, no max pool. Params: ~11.2M. FLOPs: ~1.8G. Normalization: BatchNorm.

**TRANSFORMER ARCHITECTURE:** 
ViT-Tiny (CIFAR 32x32 variant). 4x4 patch, dim 192, depth 12, heads 3. Params: ~5.7M. FLOPs: ~1.1G. Normalization: LayerNorm.
*Fallback:* Swin-Tiny. Pre-registered Practical Viability Thresholds: ViT-Tiny must achieve >65% clean accuracy on Recipe A, and be within 5% of ResNet-18, by end of Week 2. 

**TRAINING RECIPES:**
- **Recipe A (Conventional):** SGD, LR=0.1 (MultiStep), 200 epochs, WD=5e-4, Crop/Flip.
- **Recipe B (Modern):** Composite modern-regularization intervention (AdamW, Cosine LR, RandAugment, Mixup, CutMix, Label Smoothing, Random Erasing, Stochastic Depth). We do not claim to identify the contribution of individual components, only the effect of the OVERALL recipe.

**DATASET:** 
CIFAR-100 (45k train, 5k clean validation, 10k test).

**CORRUPTION BENCHMARK:** 
CIFAR-100-C (Evaluation only).

**FAIRNESS PROTOCOL:** 
Fixed architecture + fixed training budget + natural clean performance. No capacity adjustment. No early stopping. We explicitly do not claim CNN and ViT capacity is perfectly matched.

**NUMBER OF SEEDS:** 
3 final fixed seeds (42, 43, 44). Pilot runs are distinct and do not count toward the core matrix unless they exactly match final configuration and seed.

**METRICS:** 
Clean Accuracy, Standard mCE, Absolute Error Increase (AEI), ECE, Accuracy by corruption, Accuracy by severity, Error by corruption. See METRICS_SPEC.md for exact definitions.

**STATISTICAL METHOD:** 
Cautious factorial analysis reporting mean, standard deviation, confidence intervals, effect sizes, and factorial effects. 2x2 Factorial ANOVA only used if assumptions (normality, equal variance) are reasonably supported.

**CORE TRAINING RUNS:** 
12 final core training runs (2 arch x 2 recipe x 3 seeds). Frozen scope. No architectures or recipes will be added to the core design.

**WEEK-4 SAFE VERSION:** 
ResNet/ViT working, 2 recipes working, 12 core runs complete, CIFAR-100-C eval complete, clean metrics, robustness metrics, calibration pipeline implemented, reproducible saved results, preliminary factorial analysis.

**WEEK-7/8 TIMELINE:** 
All experimental results frozen by end of Week 7. Week 8 dedicated exclusively to analysis, report, figures, presentation, demo, reproducibility, and defense prep.
