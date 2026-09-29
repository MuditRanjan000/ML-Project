# RESEARCH DECISION LOG

### [2026-08-31] Initial Methodology Freeze
*   **Decision:** Frozen the core 12-run experimental matrix (ResNet-18 vs ViT-Tiny, Recipe A vs Recipe B, 3 fixed seeds).
*   **Reason:** To empirically quantify the overall effects of architecture, composite training recipes, and their interaction on robustness and calibration.

### [2026-08-31] Protocol Decision: Clean-Accuracy Matching
*   **Decision:** We will use fixed architectures, fixed training budgets, and report natural clean performance. No capacity adjustments or early stopping. We explicitly acknowledge the parameter/FLOPs mismatch (~11.2M vs ~5.7M) and use Absolute Error Increase (AEI) to isolate degradation.

### [2026-08-31] Pre-registration: ViT-Tiny Practical Viability Thresholds
*   **Decision:** If ViT-Tiny fails to achieve >65.0% clean accuracy on Recipe A, or if its accuracy is >5.0% worse than ResNet-18, we will fall back to Swin-Tiny.
*   **Reason:** ViTs are difficult to train on small datasets. These are PRACTICAL VIABILITY THRESHOLDS to ensure the base comparison isn't catastrophically broken, not scientific thresholds. Decision must be locked before the main experiment.

### [2026-08-31] Recipe B Definition
*   **Decision:** Recipe B is defined strictly as a COMPOSITE modern-regularization intervention.
*   **Reason:** The project scopes the effect of the OVERALL training recipe. We do not claim to identify individual component contributions (AdamW, Mixup, etc.).
