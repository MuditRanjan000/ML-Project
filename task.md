# Project Tasks

## Phase A-F: Planning & Methodology (Completed)
- [x] Workspace Audit
- [x] Literature Verification
- [x] Methodology Freeze
- [x] Experiment Matrix Freeze

## Phase G: Repository Setup
- [x] Initialize directory structure (`src/`, `configs/`, `scripts/`, `notebooks/`, `results/`, etc.)
- [x] Create `README.md`
- [x] Create `RESEARCH_QUESTION.md`
- [x] Create `METRICS_SPEC.md`
- [x] Create `.gitignore`
- [ ] Initialize Git repository
- [ ] Set up environment `requirements.txt`
- [ ] Implement configuration system for parameters (YAML/Argparse)

## Phase H: Minimal Baseline (Week 1-2)
- [ ] Implement CIFAR-100 dataloader (45k train, 5k val, 10k test split)
- [ ] Implement CIFAR-100-C evaluation dataloader
- [ ] Implement ResNet-18 architecture (32x32 variant)
- [ ] Implement ViT-Tiny architecture (32x32 variant)
- [ ] Implement Recipe A (Conventional) training loop
- [ ] Implement evaluation metrics (Accuracy, mCE, Absolute Error Increase)
- [ ] Implement ECE calibration metric and temperature scaling
- [ ] Verify clean baseline run of ResNet-18 on Recipe A
- [ ] Verify clean baseline run of ViT-Tiny on Recipe A (Check against ViT fallback criteria)

## Phase I: Core Experiment (Week 3-4)
- [ ] Implement Recipe B (Modern) training additions (Mixup, Cutmix, RandAugment, etc)
- [ ] Execute 12 core runs (2 Architectures x 2 Recipes x 3 Seeds)
- [ ] Log results programmatically (JSON/CSV)
- [ ] Generate preliminary factorial analysis (Effect sizes, uncertainty)
- [ ] Verify Week 4 Safe Point met

## Phase J: Evaluation System (Week 5)
- [ ] Generate detailed calibration reports (Reliability diagrams)
- [ ] Execute full CIFAR-100-C evaluation on all checkpoints

## Phase K: Analysis & Ablations (Week 6)
- [ ] Conduct Error Overlap analysis (Failure modes)
- [ ] Generate figures for final report (mCE gaps, Absolute Error Increase)
- [ ] *Optional (Tier 3)*: Fourier sensitivity analysis
- [ ] *Optional (Tier 3)*: Patchify stem ablation

## Phase L-O: Wrap Up (Week 7-8)
- [ ] Freeze all experiments
- [ ] Draft Research Report
- [ ] Draft Presentation Slides
- [ ] Create Demo
- [ ] Reproducibility audit
- [ ] Final Presentation and Oral Defense prep
