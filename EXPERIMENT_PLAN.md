# EXPERIMENT PLAN

## Overview
This document outlines the systematic execution of the 8-week empirical study comparing ResNet-18 and ViT-Tiny under distribution shift.

## Phase H: Minimal Baseline (Weeks 1-2)
*   **Objective:** Establish the data pipeline and execute **Pilot runs** to prove models can train.
*   **Tasks:**
    *   Setup CIFAR-100 (45k train, 5k clean val) and CIFAR-100-C dataloaders.
    *   Implement ResNet-18 and ViT-Tiny.
    *   Implement Recipe A.
    *   Execute **Pilot run** of ResNet-18 + Recipe A.
    *   Execute **Pilot run** of ViT-Tiny + Recipe A.
*   **Checkpoint:** Check ViT-Tiny performance against the Practical Viability Thresholds (65% minimum, within 5% of ResNet) by Friday of Week 2. Fallback to Swin-Tiny if triggered. Pilot runs do not count as core runs unless seeds/configs match perfectly.

## Phase I: Core Experiment (Weeks 3-4)
*   **Objective:** Execute the frozen 12-run core matrix.
*   **Tasks:**
    *   Implement Recipe B (Composite modern intervention).
    *   Execute 12 final core training runs using fixed final seeds (42, 43, 44).
    *   Implement full calibration pipeline (15-bin ECE, Temp Scaling on Clean Val only).
    *   Run CIFAR-100-C evaluation on all 12 core checkpoints.
    *   Calculate Clean Accuracy, Standard mCE, AEI, and ECE.
*   **Checkpoint (Week-4 Safe Point):** ResNet/ViT baselines working, Recipes A/B working, 12 core runs complete, evaluation complete, calibration pipeline implemented, results saved, preliminary factorial analysis completed.

## Phase J & K: Analysis & Ablations (Weeks 5-6)
*   **Objective:** Deepen the scientific explanation (Tier 2/3).
*   **Tasks:**
    *   Reliability Diagrams.
    *   Error Overlap analysis.
    *   (Optional Tier 3) Fourier sensitivity heatmaps, patchify stem ablation.
    *   No architectures or recipes will be added to the core design.

## Phase L-O: Wrap Up (Weeks 7-8)
*   **Objective:** Finalize deliverables.
*   **Week 7:** FREEZE all experimental results. No major new experiments.
*   **Week 8:** Focus exclusively on analysis, report writing, figures, presentation creation, demo setup, reproducibility audit, and oral defense prep.
