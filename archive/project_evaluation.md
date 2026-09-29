# Selecting a Computer Vision Final Project to Maximize Expected A-Grade Outcome
### An 8-Week Feasibility, Novelty, and Research-Quality Audit of 10 Candidate Projects

**Audience:** 2-person undergraduate team, Introduction to Machine Learning, CV-expert evaluator
**Constraint:** hard 8-week deadline, 1× A100 (availability not guaranteed), no multi-GPU
**Objective function:** maximize *expected* A-grade outcome, not theoretical research ceiling

---

## PART 0 — CRITICAL FINDING BEFORE ANYTHING ELSE

I checked all ten titles against the literature before analyzing them. Three of them are **not project ideas — they are the exact titles of published papers**:

| Project | Actual identity | Venue | Verified |
|---|---|---|---|
| **Project 8**: "Stochastic In-Context Learning for Medical Image Segmentation" | **Tyche** (Rakic, Wong, Gonzalez Ortiz, Cimini, Guttag, Dalca) | **CVPR 2024**, pp. 11159–11173 | Yes — CVPR open access + arXiv 2401.13650 |
| **Project 10**: "Scalable Interactive Segmentation with In-Context Guidance" | **MultiverSeg** (Wong, Gonzalez Ortiz, Guttag, Dalca) | **ICCV 2025** | Yes — ICCV open access + arXiv 2412.15058 |
| **Project 9**: "Lightweight Low-Light Image Enhancement via Multi-prior Retinex" | **Multinex** (Brateanu, Mu, Ancuti et al.) | **CVPR 2026** | Yes — arXiv 2604.10359 + official repo |

Projects 8 and 10 are from the **same MIT CSAIL lab** (Dalca group), and both were trained on **MegaMedical** — a curated aggregation of 53 medical segmentation datasets, >22,000 scans. You cannot reproduce either in 8 weeks on one A100. Doing so would be a compute-bound reproduction with zero contribution.

Project 9 is reproducible in principle (LOL-v1 is 485 train / 15 test pairs), but you would be reproducing a 45K-parameter model whose entire contribution is architectural micro-engineering, on a benchmark where the test set is **15 images**. That is a PSNR-chasing exercise, not a research project.

**These three are eliminated at Part 0 and only kept in the tables below for completeness.** A professor who searches your title and finds the identical paper will assume you did not do a literature review. That is a catastrophic first impression that no amount of good experiments recovers from.

The remaining seven are genuine synthesis-level ideas. They get the full treatment.

---

## PART 1 — DECONSTRUCTION OF EACH PROJECT

I use a fixed 12-field schema. "Project type" uses the taxonomy you specified.

---

### PROJECT 1 — Enhancing Few-Shot Industrial Anomaly Detection in VLMs via Spatial Adapters

**1. Core research problem.** CLIP's image encoder is trained with an image-level contrastive objective. Its patch tokens are therefore not natively aligned to the text embedding space, and its spatial localization is weak. Industrial anomaly detection (AD) needs *pixel-level* localization of small defects (a 20×20 scratch on a 1024×1024 metal part). The problem is: how do you inject spatial competence into a frozen VLM using only k∈{1,2,4,8} normal reference images?

**2. Research motivation.** Manufacturing has thousands of SKUs. Training a per-category detector (the PatchCore/PaDiM paradigm) does not scale. Zero-/few-shot generalist detectors do, but they lag behind per-category memory-bank methods on localization.

**3. Why it matters.** Cold-start inspection: a new product line has no defect history. Verified: MVTec AD's premise is exactly the cold-start problem — fit using only nominal images (Roth et al., CVPR 2022).

**4. Main research question.** Does a lightweight *spatially-structured* adapter on CLIP patch tokens improve few-shot anomaly **segmentation** more than an equal-capacity spatially-agnostic (per-token linear) adapter?

**5. Primary hypothesis.** H1: An adapter with an explicit spatial mixing operation (depthwise conv / local window attention over the patch grid) will improve pixel-AUPRO at k=4 by a margin exceeding the seed-to-seed standard deviation, relative to a parameter-matched per-token MLP adapter.

**6. Secondary hypothesis.** H2: The spatial adapter's advantage grows as defect size shrinks — i.e., the benefit is concentrated in the small-defect stratum, not uniform across the dataset.

**7. Main ML/CV concepts.** Vision-language contrastive pretraining; patch-token vs. CLS-token semantics; prompt ensembling; adapters and PEFT; memory banks and nearest-neighbour scoring; pixel-level AUROC/AUPRO/AP; score-map upsampling and smoothing.

**8. Expected methodology.** Freeze CLIP ViT-B/16. Extract intermediate-layer patch features. Train a small adapter head (≤1M params) on a *held-out set of categories* to project patch features into the text space; evaluate on unseen categories with k normal shots forming a small memory bank. Fuse text-alignment score with memory-bank distance score.

**9. What you'd build.** A CLIP feature extraction pipeline; two adapter variants (spatial vs. non-spatial); a k-shot memory bank; a score fusion module; an AUPRO evaluator; a defect-size-stratified evaluation harness.

**10. What you'd investigate.** Whether locality is the missing ingredient in VLM-based AD, and whether the gain is a genuine localization gain or an artifact of score-map smoothing.

**11. Meaningful contribution.** A parameter-matched, defect-size-stratified ablation isolating *spatial mixing* as a causal factor — plus the honest control showing how much of the "gain" a Gaussian blur on the score map already buys you. That control is genuinely underreported in this literature.

**12. Project type.** Experimental research with a small original-method component. Leans engineering-heavy.

---

### PROJECT 2 — Architectural Robustness: CNNs vs Vision Transformers Under Corruption and Domain Shift

**1. Core research problem.** The field has published contradictory claims about whether ViTs are inherently more robust than CNNs. The contradiction arises because architecture is confounded with training recipe, data scale, and model capacity. The core problem is a **causal attribution problem**, not a benchmarking problem.

**2. Research motivation.** Verified state of the literature:
- Bhojanapalli et al. (ICCV 2021) and Naseer et al. (NeurIPS 2021) report ViTs are more robust.
- **Bai, Mei, Yuille & Xie (NeurIPS 2021)** explicitly argue those conclusions came from unfair settings — mismatched scales and distinct training frameworks — and show that with a unified setup, CNNs match ViTs on adversarial robustness once they adopt the ViT training recipe. They do, however, attribute the *OOD generalization* gap to self-attention-like architecture.
- **Wang, Bai, Zhou & Xie (ICLR 2023), "Can CNNs Be More Robust Than Transformers?"** then contradicts even that, showing three purely convolutional design changes — patchifying the input, enlarging kernels, reducing activation/normalization layers — produce pure CNNs as robust as or more robust than Transformers.

So the literature has *already reversed itself twice*. That is a real, live, unsettled question and it is exactly the kind of thing a CV professor finds interesting.

**3. Why it matters.** If robustness is recipe-driven, practitioners should change their augmentation pipeline (cheap). If it is architecture-driven, they must change their backbone (expensive). The two prescriptions differ completely.

**4. Main research question.** Under a *matched in-distribution accuracy* protocol, how much of the CNN–ViT corruption-robustness gap is attributable to architecture, and how much to the training recipe — and does the same decomposition hold for **calibration** under shift as for accuracy?

**5. Primary hypothesis.** H1: When ResNet and ViT-family models are matched on clean test accuracy and trained with an identical recipe, the mean corruption error (mCE) gap will shrink to a small fraction of the gap observed under each family's default recipe. The recipe explains most of the accuracy gap.

**6. Secondary hypothesis.** H2: Calibration under shift **dissociates** from accuracy under shift. The architectural effect on Expected Calibration Error at high corruption severity will *not* be eliminated by recipe matching, even when the accuracy effect is. Grounded in Minderer et al. (NeurIPS 2021), who found non-convolutional families (ViT, MLP-Mixer) degrade more slowly in calibration on ImageNet-C and argued architecture is a major determinant of calibration.

**7. Main ML/CV concepts.** Inductive bias (locality, translation equivariance, weight sharing) vs. global self-attention; receptive field and effective patch size; data augmentation as an implicit prior; batch-norm vs. layer-norm statistics under shift; corruption robustness benchmarks; calibration, ECE, temperature scaling; Fourier-domain sensitivity analysis.

**8. Expected methodology.** Two-tier design.
*Tier A (evaluation-only, no training):* evaluate a matched set of pretrained `timm` models on a corruption benchmark; report accuracy, mCE, ECE.
*Tier B (controlled, from-scratch):* full factorial — {ResNet-18, ViT-Tiny/Swin-Tiny} × {minimal recipe, DeiT-style recipe, DeiT+AugMix} × 3 seeds — all trained on the same dataset, all evaluated on the same corruption suite. Then variance decomposition of mCE across the factors.

**9. What you'd build.** A single unified training harness where architecture and recipe are independent config flags (this is the scientific heart — one codebase, one dataloader, one eval); a corruption evaluation suite; an ECE/reliability-diagram module; a Fourier heatmap sensitivity analyzer (Yin et al., NeurIPS 2019 — official method, unofficial PyTorch implementations exist).

**10. What you'd investigate.** Whether "ViTs are more robust" survives confound control, and *why* — via frequency sensitivity profiles that explain which corruption families each architecture is vulnerable to.

**11. Meaningful contribution.** (a) A matched-accuracy fair-comparison protocol at a scale reproducible by two students; (b) an explicit variance decomposition attributing mCE to architecture vs. recipe vs. seed; (c) the accuracy/calibration dissociation result, which is under-reported because most architecture comparisons report accuracy only; (d) Fourier-sensitivity profiles as a *mechanistic* explanation rather than a leaderboard.

**12. Project type.** Experimental research (controlled/confound-isolating). This is the classic form of a strong empirical ML paper.

---

### PROJECT 3 — From Memory Banks to Foundation Models: Zero-/Few-Shot IAD with Hybrid Score Fusion

**1. Core research problem.** Two families of anomaly scores exist: (i) **distance-to-nominal-features** (memory bank: SPADE, PaDiM, PatchCore) and (ii) **semantic mismatch with text** (CLIP-based: WinCLIP, AnomalyCLIP, APRIL-GAN). They fail on different things. When should you fuse, and by what rule?

**2. Research motivation.** Verified anchors: PatchCore reaches up to **99.6% image AUROC on MVTec AD** (Roth et al., CVPR 2022) — the benchmark is saturated in the full-shot regime. WinCLIP (Jeong et al., CVPR 2023) reports ~91.8% zero-shot image AUROC on MVTec AD with no task-specific training. The *few-shot* regime (k≤4) is where both are weak and where the headroom is.

**3. Why it matters.** The practical question in industry is "I have 4 good images of a new part — what do I run?" Nobody has given a defensible answer that spans both paradigms with a principled fusion rule.

**4. Main research question.** Does *adaptive* fusion of a memory-bank score and a language-alignment score — weighted by a per-image reliability estimate — outperform the best single score and a fixed-weight fusion, in the k∈{1,2,4} regime?

**5. Primary hypothesis.** H1: The two score families are **complementary**, and the degree of complementarity is predictable from a cheap per-image statistic (e.g., nearest-neighbour distance in the k-shot bank ≈ "is this image well-covered by my references?"). Reliability-weighted fusion beats fixed-weight fusion at k=1 and the advantage decays as k grows.

**6. Secondary hypothesis.** H2: Replacing the ImageNet-supervised WideResNet-50 backbone in PatchCore with a self-supervised foundation backbone (DINOv2) closes more of the few-shot gap than adding the CLIP text score does — i.e., **representation quality dominates score fusion**.

H2 is the interesting one. If it holds, the paper's message is "stop building fusion rules, fix your features." That is a genuinely useful negative-ish result.

**7. Main ML/CV concepts.** Coreset subsampling and greedy k-center; Mahalanobis vs. kNN scoring; patch-feature aggregation across layers; CLIP prompt ensembling; score normalization and calibration before fusion; AUROC vs. AUPRO vs. AP under extreme class imbalance.

**8. Expected methodology.** Reimplement (or use `anomalib`) PatchCore + PaDiM + a WinCLIP-style zero-shot scorer. Cache all features once. Grid the fusion rules: (a) none, (b) fixed-α linear, (c) rank fusion, (d) reliability-gated. Ablate backbone independently of fusion. Evaluate on MVTec AD, transfer to VisA.

**9. What you'd build.** A feature cache; three scorers; four fusion rules; a k-shot sampler with fixed seeds; an AUPRO implementation; a per-category breakdown and per-defect-type error analysis.

**10. What you'd investigate.** Which of {backbone, score type, fusion rule, k} actually moves the needle — an attribution study, not a leaderboard climb.

**11. Meaningful contribution.** A clean 2×2 factorial (backbone × score-family) that separates representation effects from fusion effects. This is the missing control in most AD fusion papers.

**12. Project type.** Experimental research + systematic reproduction. Honest, defensible, low-glamour.

---

### PROJECT 4 — To Fine-tune or to Prompt: PEFT of ViTs Under Label Scarcity and Distribution Shift

**1. Core research problem.** Given a frozen pretrained ViT, n labeled examples, and a test distribution that differs from train, which adaptation method wins: linear probe, BitFit, LoRA, VPT, adapters, or full fine-tuning? And does the ranking *change* between in-distribution and out-of-distribution evaluation?

**2. Research motivation.** Verified anchor: **Kumar, Raghunathan, Jones, Ma & Liang (ICLR 2022)** show full fine-tuning beats linear probing in-distribution but *underperforms it OOD* when pretrained features are good and the shift is large — across 10 shift datasets, fine-tuning averaged ~2% better ID and ~7% worse OOD. Their explanation is feature distortion, and their fix (LP-FT: linear-probe first, then fine-tune) gets ~1% better ID and ~10% better OOD than full FT.

The natural extension: PEFT methods are *structurally* constrained not to distort features much. Do they inherit linear probing's OOD robustness while keeping fine-tuning's ID accuracy?

**3. Why it matters.** This is the single most common decision an ML practitioner makes with a foundation model, and the standard advice ignores distribution shift entirely.

**4. Main research question.** Does the ID-vs-OOD ranking of PEFT methods invert as a function of (a) label budget and (b) shift severity — and is the inversion point predictable from a measure of feature distortion?

**5. Primary hypothesis.** H1: There exists a crossover regime — at small n and large shift, low-capacity adaptation (linear probe, BitFit) beats high-capacity adaptation (full FT, high-rank LoRA) OOD, even while losing ID. The crossover moves monotonically with n.

**6. Secondary hypothesis.** H2: OOD degradation is monotonically predicted by a **feature-drift metric** (e.g., CKA or mean cosine distance between pre- and post-adaptation penultimate features on a held-out unlabeled set), independent of which PEFT family produced the drift. If true, this gives a *label-free diagnostic* for "have I over-adapted?"

H2 is the actual contribution. Ranking methods is a benchmark; predicting the ranking from a computable quantity is research.

**7. Main ML/CV concepts.** Transfer learning; PEFT (LoRA low-rank updates, VPT prompt tokens, BitFit bias-only, adapters); feature distortion; representation similarity (CKA); ID/OOD generalization gap; few-shot evaluation variance.

**8. Expected methodology.** Freeze a ViT-B/16 (CLIP or DINOv2 or ImageNet-21k supervised — pick ONE). Adapt with 5 methods × {n=1,5,25,full} × 3 seeds on a source domain. Evaluate on the source test set and on 2–3 shifted targets. Compute CKA drift for every run. Regress OOD gap on drift.

**9. What you'd build.** A single adaptation harness with swappable PEFT modules (peft/timm make this tractable); a CKA implementation; a domain-shift evaluation loop; a variance-aware results table.

**10. What you'd investigate.** Whether "which PEFT method" is even the right question, or whether it collapses to "how much did you move the features."

**11. Meaningful contribution.** The drift-predicts-OOD-degradation regression, with the falsifiable claim that it generalizes across PEFT families. Plus a low-shot, shift-aware protocol.

**12. Project type.** Experimental research with a diagnostic-metric contribution.

**Important honesty note.** A 2024 unifying study of PEFT in visual recognition (LoRA, VPT, adapters + ~10 more, on VTAB-1K plus ImageNet domain-shift variants) already found that representative PEFT methods perform *similarly* on VTAB-1K once properly implemented and hyperparameter-tuned. You must cite this and position against it — your differentiator is the drift metric and the low-shot × shift-severity crossover surface, not the ranking itself.

---

### PROJECT 5 — Representation or Acquisition? Calibration-Gated Active Learning in the VFM Era

**1. Core research problem.** Classical active learning (AL) wisdom says uncertainty sampling fails in low-budget regimes (the "cold start") because the model's uncertainty is unreliable. Foundation-model features may have removed the cold start entirely. If so, most of the AL literature's core finding is obsolete.

**2. Research motivation.** Verified: Hacohen, Dekel & Weinshall (ICML 2022) introduced TypiClust and showed low- and high-budget regimes call for opposite strategies, with uncertainty methods no better than random at low budget. But **Gupte, Aklilu, Nirschl & Yeung-Levy (TMLR 2024), "Revisiting Active Learning in the Era of Vision Foundation Models"** report that with DINOv2/OpenCLIP features and controlled random initialization, uncertainty sampling is competitive with TypiClust as early as the *second* AL iteration and beats it later — a direct deviation from prior literature. Their code is public.

**3. Why it matters.** Annotation budget is the binding constraint in most applied CV. Getting the acquisition function wrong wastes money.

**4. Main research question.** Is the residual benefit of uncertainty-based acquisition over random sampling predicted by the model's **calibration** at that budget — and can a calibration-gated switch (use diversity while ECE is high, switch to uncertainty once ECE drops below τ) recover the best of both without knowing the budget in advance?

**5. Primary hypothesis.** H1: The AL round at which uncertainty sampling begins to outperform diversity sampling coincides with the round at which held-out ECE of the linear probe falls below a threshold; a calibration-gated switch matches or beats the best fixed strategy across budgets.

**6. Secondary hypothesis.** H2: Backbone quality dominates acquisition function. The accuracy spread across {DINOv2, CLIP, ImageNet-supervised ResNet} at a fixed budget exceeds the spread across all acquisition functions within any single backbone.

**7. Main ML/CV concepts.** Active learning; uncertainty (entropy, margin, BALD) vs. diversity (coreset, TypiClust, ProbCover); calibration and ECE; linear probing on frozen features; budget regimes; high-variance experimental design requiring many seeds.

**8. Expected methodology.** Precompute frozen embeddings once (this is the key efficiency trick — the whole study then runs on cached vectors and a linear head, i.e. **minutes per AL run on CPU/GPU**). Run 6 acquisition functions × 3 backbones × 8 budget rounds × 5 seeds. Track accuracy and ECE per round. Fit the gating threshold on one dataset, test transfer to another.

**9. What you'd build.** An embedding cache; an AL loop; six acquisition functions; a calibration tracker; a gated-switch policy; seed-variance plots with confidence bands.

**10. What you'd investigate.** Whether AL research conclusions are backbone-dependent artifacts.

**11. Meaningful contribution.** Calibration as an *observable, budget-free trigger* for strategy switching. Compare against the known TCM heuristic (TypiClust→Margin), which switches on a hand-set budget rather than a measured signal — that is a clean, honest delta.

**12. Project type.** Experimental research. Solid, but the ground is well-trodden and the strongest recent paper is essentially your project with more compute.

---

### PROJECT 6 — Persistent Errors as Signals for Mitigating Visual Shortcut Learning

**1. Core research problem.** Models exploit spurious features (backgrounds, texture, co-occurring objects) and fail on minority groups where the shortcut breaks. Group-robust methods need group annotations. Annotation-free methods infer the minority group from a first model's *errors*. The open question: **which errors?** A single ERM run's error set is noisy — it mixes genuine shortcut-violating examples with label noise, hard-but-not-spurious examples, and pure stochastic misclassifications.

**2. Research motivation.** Verified anchors:
- **Sagawa et al. (ICLR 2020)** — GroupDRO, the annotation-requiring oracle.
- **Liu et al. (ICML 2021), JTT** — train ERM briefly, upweight what it got wrong, retrain. Recovers ~73% of the ERM→GroupDRO worst-group gap with no training group labels; +15.9% mean worst-group accuracy over ERM at ~3.7% average-accuracy cost. Critically, the JTT authors themselves found the *group composition* of the error set matters but the *specific examples* do not (swapping error-set examples for same-group examples cost only 0.7% worst-group accuracy). That is a direct invitation to ask what signal actually identifies the right group.
- **Kirichenko, Izmailov & Wilson (ICLR 2023, top-25%), DFR** — networks that appear to rely on backgrounds still *learn* core features; retraining only the last layer on a small group-balanced set matches or beats SOTA at a fraction of the cost, in minutes on one GPU.

**3. Why it matters.** Shortcut learning is the canonical failure mode of deployed vision systems and is visually demonstrable, which matters for your presentation.

**4. Main research question.** Does **error persistence** — misclassification consistently across independent seeds and/or across training epochs — identify shortcut-violating examples more precisely than a single-run error set, and does higher precision translate into better worst-group accuracy?

**5. Primary hypothesis.** H1: Persistence-filtered error sets have higher precision *and* recall for true minority-group membership than single-run JTT error sets at matched set size; and a JTT-style reweighting on the persistent set yields higher worst-group accuracy than standard JTT at equal compute budget.

**6. Secondary hypothesis.** H2 (the interesting fallback): The gain from persistence saturates quickly — 2 seeds captures nearly all of it — because error-set *composition* rather than *identity* drives the benefit, consistent with the JTT authors' own ablation. If H2 holds, the contribution is a compute-efficiency result: "you need N seeds, not more, and here's why."

**7. Main ML/CV concepts.** Spurious correlation and simplicity bias; worst-group vs. average accuracy; group DRO; two-stage reweighting; last-layer retraining; feature-vs-classifier attribution of bias; saliency/attention visualization; precision/recall of an inferred group label.

**8. Expected methodology.** Waterbirds + CelebA. Train k ERM models with different seeds. Define persistence score per training example. Build error sets at matched sizes under several definitions. Retrain with upweighting. Compare to ERM, JTT (1 seed), DFR, GroupDRO (oracle). Measure error-set precision/recall against the *known* group labels (available in these datasets — that is the whole reason to use them).

**9. What you'd build.** An ERM training loop with seed control; a persistence scorer; an error-set constructor with a size-matching guarantee; four baselines; a group-precision/recall evaluator; a background-swap qualitative viewer.

**10. What you'd investigate.** Whether the "identify then upweight" family is bottlenecked by identification quality at all, or by something else entirely (e.g., the classifier head, per DFR).

**11. Meaningful contribution.** A measurement result — the precision/recall of inferred error sets against ground-truth groups as a function of the identification signal — that the literature reports only indirectly. Plus the compute-vs-precision curve.

**12. Project type.** Experimental research with an original (modest, well-scoped) method component. Strongest novelty-per-unit-risk of the seven.

---

### PROJECT 7 — When Adaptation Hurts: Reliability-Aware Test-Time Adaptation Under Changing Corruptions

**1. Core research problem.** Test-time adaptation (TTA) updates a model online on unlabeled test data. It sometimes helps a lot and sometimes destroys the model. The problem is deciding, without labels, *whether to adapt at all* on a given batch.

**2. Research motivation.** Verified:
- **TENT** (Wang et al., ICLR 2021 spotlight) — minimize prediction entropy, update BN affine parameters only.
- **EATA** (Niu et al., ICML 2022) — entropy-based sample filtering + redundancy filtering + Fisher regularization.
- **SAR** (Niu et al., ICLR 2023 oral) — filter high-entropy samples and use sharpness-aware minimization to avoid collapse.
- **Zhao, Liu, Alahi & Lin (ICML 2023), "On Pitfalls of Test-Time Adaptation" (TTAB)** — benchmark of 10 algorithms; three documented pitfalls: hyperparameter/model selection is very hard due to online batch dependency; performance depends strongly on pretrained-model quality; several shift classes defeat current methods. Reported failures include BN-adapt at 77.8% error and TENT above 76% error under severe label shift, and TENT/BN-adapt failing to improve under non-stationary shift even with good model selection.

**3. Why it matters.** TTA is deployed in real systems. "When does it hurt" is more actionable than "here's +0.3 mCE."

**4. Main research question.** Can a per-batch, label-free reliability signal decide *whether* to apply an adaptation update, and does such gating convert TTA from high-variance to reliably non-harmful across corruption types, severities, and batch compositions?

**5. Primary hypothesis.** H1: A gate based on a distribution-level statistic (e.g., BN-statistic distance between the batch and source running statistics, or predictive-entropy dispersion within the batch) predicts the *sign* of the adaptation benefit with AUROC well above chance, and gated TTA achieves strictly lower worst-case error than ungated TENT while retaining most of its average-case gain.

**6. Secondary hypothesis.** H2: The gate's value is concentrated in adversarial-for-TTA conditions — small batches, class-imbalanced batches, mild shift where the source model was already fine — and vanishes in the standard large-batch, uniform-label, severity-5 setting that the literature usually reports. That is, **the standard evaluation protocol hides the problem.**

H2 is a protocol critique and is the most defensible part of this project.

**7. Main ML/CV concepts.** Covariate shift; BN statistics and their role in adaptation; entropy minimization and its collapse dynamics; online/episodic protocols; confirmation bias in self-training; error accumulation in continual adaptation.

**8. Expected methodology.** CIFAR-10-C / CIFAR-100-C with a fixed source model. Implement source-only, BN-adapt, TENT; add EATA-style filtering as the strong modern baseline. Sweep batch size, label-distribution skew, severity, and corruption ordering (episodic vs. continual). Fit the gate on a subset of corruptions, test on held-out corruptions.

**9. What you'd build.** A TTA harness with episodic reset control; three to four TTA algorithms; a stream simulator with configurable batch composition; a gate module; a "sign-prediction AUROC" evaluator.

**10. What you'd investigate.** Whether TTA's reported gains survive realistic stream conditions, and whether abstention is a better lever than better loss functions.

**11. Meaningful contribution.** A harm-conditions map (which stream properties flip the sign of TTA's benefit) plus a gate evaluated on *held-out corruption types*, not the ones it was tuned on.

**12. Project type.** Experimental research with a small method component. Scientifically excellent; operationally fragile.

---

### PROJECTS 8, 9, 10 — brief deconstruction (eliminated at Part 0)

**Project 8 (= Tyche, CVPR 2024).** Problem: in-context medical segmentation that outputs a *set* of plausible masks rather than one, capturing inter-annotator disagreement. Contribution in the real paper: a convolution block enabling interaction among the predicted candidates, and in-context test-time augmentation for stochasticity. Two variants (train-time and inference-time stochasticity). Requires MegaMedical-scale training. What you would actually do: download weights and run inference. Project type: **research reproduction, compute-blocked**.

**Project 9 (= Multinex, CVPR 2026).** Problem: LLIE at edge-deployable parameter counts. Contribution in the real paper: residual Retinex decomposition, analytic multi-representation illumination/color priors, lightweight learnable fusion; 45K and 0.7K parameter versions. Datasets: LOL-v1/v2-real/v2-synthetic + no-reference sets (DICM, MEF, NPE, LIME). Project type: **research reproduction / engineering**. Feasible but the contribution is architectural micro-optimization on a 15-image test set.

**Project 10 (= MultiverSeg, ICCV 2025).** Problem: amortizing annotation effort across a dataset — as the user labels more images, those become context and fewer interactions are needed per new image. Reported: 53% fewer scribble steps and 36% fewer clicks vs. a SOTA interactive method to reach 90% Dice on unseen tasks. Requires training on the same MegaMedical-scale corpus with simulated interaction loops. Project type: **research reproduction, compute-blocked**.

---

## PART 2 — RESEARCH LANDSCAPE AND GAP STATEMENTS

Everything in the "Verified" column below was checked against the paper's own abstract/proceedings page during this analysis. Anything I could not verify, I say so.

### Venue quality note (CORE / community consensus)
CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR are all **CORE A\*** and are the top tier for this work. AAAI is A\*; ACM MM, WACV, BMVC, MICCAI are **A**. TMLR is a respected journal without a CORE rank (it is new); treat a TMLR paper as roughly conference-A\*-adjacent in rigour but note it is not a CORE-ranked venue if your professor asks. TPAMI and IJCV are the flagship journals.

---

### Landscape for PROJECT 1 (VLM few-shot AD)

| Paper | Authors | Yr | Venue | Contribution | Relation to project |
|---|---|---|---|---|---|
| WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation | Jeong, Zou, Kim, Zhang, Ravichandran, Dabeer | 2023 | CVPR (A\*) | Window/patch/image-level CLIP feature ensemble aligned with a compositional state-word + template text ensemble; ~91.8% zero-shot image AUROC on MVTec AD | The direct predecessor; your baseline |
| AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot AD | Zhou, Pang, Tian, He, Chen | 2024 | ICLR (A\*) | Learns object-agnostic prompts capturing generic normality/abnormality; strong transfer across many datasets | Shows prompt learning already covers a large slice of the design space |
| APRIL-GAN (VAND challenge report) | Chen, Han, Zhang | 2023 | CVPR-W | Adds *extra linear layers* mapping image features into the joint embedding space + multi-shot memory banks; 1st place zero-shot track | **This is essentially "adapters on CLIP for AD" already** |
| AdaCLIP | Cao, Zhang, Frittoli, Cheng, Shen, Boracchi | 2024 | ECCV (A\*) | Hybrid learnable prompts for zero-shot AD | Further crowds the adapter/prompt space |
| VCP-CLIP | Qu, Tao, Prasad, Shen, Zhang, Gong, Ding | 2024 | (arXiv/ECCV-line) | Visual context prompting for zero-shot anomaly segmentation | Same |
| InCTRL (Toward Generalist AD) | Zhu, Pang et al. | 2024 | CVPR (A\*) | In-context residual learning for generalist few-shot AD | The current strong few-shot generalist baseline |

**What is already known.** Adapting CLIP to AD via linear projection layers, learnable prompts, and memory banks is a solved-in-outline problem with at least five A\*-venue entries in 24 months.
**Strongest existing approaches.** AnomalyCLIP / AdaCLIP for zero-shot; InCTRL for few-shot generalist.
**Remaining limitations.** Adapter *architecture* is under-ablated — most papers change several things at once. Defect-size-conditional analysis is rare. Score-map post-processing (Gaussian smoothing) is a known confound that few papers isolate.

> **RESEARCH GAP STATEMENT (P1).** Existing CLIP-for-AD adapters are compared as whole systems; no published study isolates *spatial mixing capacity in the adapter* at matched parameter count, or reports whether the resulting gains survive a Gaussian-smoothing control and are concentrated in small defects.

Honest assessment: this is a **narrow, real gap**, but it sits inside the most crowded subfield on your list, and the reviewer-equivalent reaction ("didn't APRIL-GAN already add linear layers?") is a live risk in oral defense.

---

### Landscape for PROJECT 2 (CNN vs ViT robustness)

| Paper | Authors | Yr | Venue | Contribution | Relation to project |
|---|---|---|---|---|---|
| Benchmarking NN Robustness to Common Corruptions and Perturbations | Hendrycks, Dietterich | 2019 | ICLR (A\*) | ImageNet-C / CIFAR-10-C / CIFAR-100-C / Tiny-ImageNet-C; 15 corruptions × 5 severities; mCE metric normalized by AlexNet | Your benchmark and your metric |
| ImageNet-trained CNNs are biased towards texture | Geirhos et al. | 2019 | ICLR (A\*) | Texture-vs-shape bias; increasing shape bias improves robustness | Mechanistic framing for your discussion |
| A Fourier Perspective on Model Robustness | Yin, Gontijo Lopes, Shlens, Cubuk, Gilmer | 2019 | NeurIPS (A\*) | Fourier heatmap sensitivity analysis; explains why Gaussian aug / adv. training help high-frequency corruptions and hurt low-frequency ones (e.g. fog, contrast) | **Your explanatory tool** — turns a benchmark into a mechanism study |
| Understanding Robustness of Transformers for Image Classification | Bhojanapalli, Chakrabarti, Glasner, Li, Unterthiner, Veit | 2021 | ICCV (A\*) | Early systematic ViT robustness study | Thesis position |
| Intriguing Properties of Vision Transformers | Naseer, Ranasinghe, Khan, Hayat, Shahbaz Khan, Yang | 2021 | NeurIPS (A\*) | ViT robustness to occlusion, patch permutation, distribution shift | Thesis position |
| **Are Transformers More Robust Than CNNs?** | Bai, Mei, Yuille, Xie | 2021 | NeurIPS (A\*) | Prior comparisons were unfair (different scales, different frameworks); with a unified setup CNNs match ViTs adversarially once given ViT training recipes; OOD generalization advantage attributed to self-attention itself | **Antithesis — your core motivating paper** |
| Revisiting the Calibration of Modern Neural Networks | Minderer, Djolonga, Romijnders, Hubis, Zhai, Houlsby, Tran, Lucic | 2021 | NeurIPS (A\*) | 180 models × 16 families × 79 datasets; non-convolutional families (ViT, MLP-Mixer) among the best calibrated; calibration decays more slowly under ImageNet-C shift; size/pretraining don't fully explain it → architecture is a major determinant | **Your second axis (H2)** |
| **Can CNNs Be More Robust Than Transformers?** | Wang, Bai, Zhou, Xie | 2023 | ICLR (A\*) | Three simple conv design changes (patchify input, larger kernels, fewer activation/norm layers) yield pure CNNs as robust as or more robust than Transformers | **Synthesis — the current frontier** |
| Understanding The Robustness in Vision Transformers (FAN) | Zhou, Yu, Xie, Xiao, Anandkumar, Feng, Alvarez | 2022 | ICML (A\*) | Attributes ViT robustness to emergent mid-level visual grouping; proposes fully-attentional networks | Mechanistic alternative hypothesis |

**What is already known.** Raw comparisons are confounded. Recipe matters enormously. Architecture matters too, but the specific architectural ingredients are contested (self-attention vs. patchification vs. kernel size vs. normalization density).
**Strongest existing approaches.** Bai et al.'s unified-training protocol; Wang et al.'s ingredient-level ablation.
**Remaining limitations.** (i) Both frontier papers matched on *parameter scale and recipe* but not on *achieved clean accuracy* — a residual confound, since a model that is 2 points worse in-distribution is not comparable OOD. (ii) Neither jointly reports **calibration** under shift alongside accuracy. (iii) The variance attributable to *random seed* is almost never reported in these comparisons, so "gaps" of 1–2 mCE points are of unknown significance.

> **RESEARCH GAP STATEMENT (P2).** The CNN-vs-ViT robustness literature has not reported a single controlled study that (a) matches models on achieved in-distribution accuracy rather than parameter count, (b) treats architecture and training recipe as crossed factors with replicate seeds so that the mCE gap can be variance-decomposed, and (c) reports corruption robustness and calibration-under-shift jointly to test whether the two dissociate.

This gap is **real, small enough to close in 8 weeks, and directly answerable with cheap experiments.** That combination is rare.

---

### Landscape for PROJECT 3 (memory banks → foundation models, AD)

| Paper | Authors | Yr | Venue | Contribution | Relation |
|---|---|---|---|---|---|
| MVTec AD dataset | Bergmann, Batzner, Fauser, Sattlegger, Steger | 2021 (dataset 2019) | IJCV | 15 categories, ~5,354 images, pixel-level masks | Primary dataset |
| PaDiM | Defard et al. | 2021 | ICPR | Patch-wise multivariate Gaussian + Mahalanobis distance | Strong standard baseline |
| **PatchCore** (Towards Total Recall) | Roth, Pemula, Zepeda, Schölkopf, Brox, Gehler | 2022 | CVPR (A\*) | Coreset-subsampled memory bank of nominal patch features; **up to 99.6% image AUROC**, 98.1–98.2% pixel AUROC, 93.5 PRO on MVTec AD; competitive in the few-sample regime | The method to beat / build on |
| VisA (SPot-the-Difference) | Zou, Jeong, Pemula, Zhang, Dabeer | 2022 | ECCV (A\*) | 10,821 images (9,621 normal / 1,200 anomalous), 12 objects, 3 domains, image + pixel labels; harder than MVTec | Secondary/generalization dataset |
| WinCLIP | Jeong et al. | 2023 | CVPR (A\*) | See P1 | Language-side score |
| DINOv2 | Oquab et al. | 2023/24 | TMLR | Self-supervised features with strong dense-prediction properties | Backbone swap arm |

**What is known.** Memory-bank methods dominate the full-shot regime to the point of saturation; VLM methods dominate the zero-shot regime; the few-shot middle is contested.
**Limitations.** Fusion is usually fixed-weight and reported only as a system-level number. Backbone and fusion are almost never ablated orthogonally. MVTec's saturation means small improvements are not statistically meaningful.

> **RESEARCH GAP STATEMENT (P3).** No published study crosses *backbone family* (ImageNet-supervised vs. self-supervised foundation) with *score family* (memory distance vs. language alignment) in a single factorial on the k≤4 regime, so the field cannot say whether few-shot AD progress comes from better representations or from better score fusion.

---

### Landscape for PROJECT 4 (PEFT under scarcity + shift)

| Paper | Authors | Yr | Venue | Contribution | Relation |
|---|---|---|---|---|---|
| **Fine-Tuning can Distort Pretrained Features and Underperform OOD** | Kumar, Raghunathan, Jones, Ma, Liang | 2022 | ICLR (A\*) | FT beats LP in-distribution but loses OOD when features are good and shift is large (~+2% ID, −7% OOD across 10 shift datasets); theory of feature distortion; LP-FT fix (~+1% ID, +10% OOD vs FT) | **Core motivating paper** |
| Visual Prompt Tuning (VPT) | Jia et al. | 2022 | ECCV (A\*) | Learnable prompt tokens prepended in ViT layers; <1% params | Baseline arm |
| LoRA | Hu et al. | 2022 | ICLR (A\*) | Low-rank weight updates | Baseline arm |
| BitFit | Ben-Zaken, Goldberg, Ravfogel | 2022 | ACL | Bias-only tuning | Cheapest PEFT arm |
| Lessons and Insights from a Unifying Study of PEFT in Visual Recognition | (Ohio State et al.) | 2024 | arXiv (later venue not verified here) | LoRA/VPT/adapter + ~10 methods on VTAB-1K with systematic hyperparameter tuning; **methods perform similarly when properly implemented and tuned**; includes ImageNet domain-shift robustness study | **Closest prior work — must be cited and positioned against** |

**Limitations.** The unifying study tunes hyperparameters *per method* and evaluates mostly at VTAB-1K's fixed 1,000-example budget; the *interaction* between label budget and shift severity is not mapped as a surface, and no label-free predictor of over-adaptation is offered.

> **RESEARCH GAP STATEMENT (P4).** Existing PEFT comparisons report which method wins; none provides a **label-free, computable predictor** (feature-drift magnitude) of how much OOD accuracy an adaptation will cost, validated to hold across PEFT families and across the label-budget × shift-severity plane.

---

### Landscape for PROJECT 5 (AL in the VFM era)

| Paper | Authors | Yr | Venue | Contribution | Relation |
|---|---|---|---|---|---|
| Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets (TypiClust) | Hacohen, Dekel, Weinshall | 2022 | ICML (A\*) | Theory + method: low budget wants typical+diverse, high budget wants uncertain; uncertainty methods ≈ random at low budget | Classical position |
| ProbCover | Yehuda, Dekel, Hacohen, Weinshall | 2022 | — | Max-coverage selection in SSL embedding space | Diversity baseline |
| Coreset / k-center | Sener, Savarese | 2018 | ICLR (A\*) | Core-set selection for AL | Diversity baseline |
| **Revisiting Active Learning in the Era of Vision Foundation Models** | Gupte, Aklilu, Nirschl, Yeung-Levy | 2024 | TMLR | With DINOv2/OpenCLIP frozen features, uncertainty sampling is competitive with TypiClust by AL iteration 2 and beats it later, once random initialization is controlled; proposes a dropout-uncertainty + diversity strategy; public code | **This is your project, already done, with more compute** |
| Bridging Diversity and Uncertainty in AL with SSL Pre-Training (TCM) | Doucet, Estermann, Aczel, Wattenhofer | 2024 | ICLR-W | Heuristic switch: TypiClust first, then Margin | Your closest method comparator |

> **RESEARCH GAP STATEMENT (P5).** Existing switch heuristics between diversity and uncertainty acquisition are triggered by a hand-chosen budget threshold; no work tests whether a *measured* signal — held-out calibration error of the current probe — predicts the switch point and transfers across datasets and backbones.

Honest assessment: this gap is genuine but **thin**, and the TMLR paper occupies most of the interesting territory.

---

### Landscape for PROJECT 6 (shortcut learning / persistent errors)

| Paper | Authors | Yr | Venue | Contribution | Relation |
|---|---|---|---|---|---|
| Shortcut Learning in Deep Neural Networks | Geirhos, Jacobsen, Michaelis, Zemel, Brendel, Bethge, Wichmann | 2020 | Nature Machine Intelligence | Conceptual framing of shortcut learning | Framing |
| Distributionally Robust Neural Networks (GroupDRO) | Sagawa, Koh, Hashimoto, Liang | 2020 | ICLR (A\*) | Worst-group loss minimization with known groups; introduced the Waterbirds construction | Oracle upper bound |
| Learning from Failure (LfF) | Nam, Cha, Ahn, Lee, Shin | 2020 | NeurIPS (A\*) | Deliberately biased first model identifies bias-conflicting samples; interleaved training | Prior identify-then-reweight method |
| **Just Train Twice (JTT)** | Liu, Haghgoo, Chen, Raghunathan, Koh, Sagawa, Liang, Finn | 2021 | ICML (A\*) **Oral** | Two-stage: brief ERM → upweight its errors → retrain. +15.9% mean worst-group accuracy over ERM at ~3.7% average-accuracy cost; closes ~73% of the ERM→GroupDRO gap without training group labels. **Own ablation: error-set group composition matters, specific examples do not (−0.7% when swapped within group)** | **Your primary baseline and the source of your research question** |
| **Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations (DFR)** | Kirichenko, Izmailov, Wilson | 2023 | ICLR (A\*) **Oral / top-25%** | Networks still learn core features even when relying on spurious ones; retraining only the last layer matches or beats SOTA with far lower cost — minutes on one GPU; also reduces background/texture reliance on ImageNet models | **Your strong modern baseline and a rival explanation** |
| Towards Last-Layer Retraining for Group Robustness with Fewer Annotations | LaBonte, Muthukumar, Kumar | 2023 | NeurIPS (A\*) | Reduces DFR's annotation requirement | Extension baseline |

**What is known.** Identify-then-reweight works without group labels; last-layer retraining works even better and far cheaper; the representation is usually not the problem, the head is.
**Limitations.** The *identification* step is evaluated only indirectly, through downstream worst-group accuracy. The precision/recall of the inferred error set against ground-truth groups, as a function of the identification signal and its compute cost, is not systematically reported.

> **RESEARCH GAP STATEMENT (P6).** Annotation-free group-robustness methods infer minority membership from a single training run's errors, but no study measures how the *precision and recall of that inference* scale with error-persistence evidence (seeds × epochs), nor whether identification quality — as opposed to error-set group composition — is the binding constraint on downstream worst-group accuracy.

This is a **measurement gap**, which is the safest kind of gap for undergraduates: you get a result even if your method loses.

---

### Landscape for PROJECT 7 (TTA)

| Paper | Authors | Yr | Venue | Contribution | Relation |
|---|---|---|---|---|---|
| TENT: Fully Test-Time Adaptation by Entropy Minimization | Wang, Shelhamer, Liu, Olshausen, Darrell | 2021 | ICLR (A\*) **Spotlight** | Online entropy minimization over BN affine parameters; one update per batch | Core baseline |
| EATA | Niu, Wu, Zhang, Chen, Zheng, Zhao, Tan | 2022 | ICML (A\*) | Reliable (low-entropy) + non-redundant sample filtering; Fisher anti-forgetting regularizer | Strong modern baseline |
| SAR: Towards Stable TTA in Dynamic Wild World | Niu, Wu, Zhang, Wen, Chen, Zhao, Tan | 2023 | ICLR (A\*) **Oral** | High-entropy filtering + sharpness-aware minimization to avoid collapse | Strong modern baseline |
| CoTTA | Wang, Fink, Van Gool, Dai | 2022 | CVPR (A\*) | Continual TTA with weight-averaged/augmentation-averaged pseudo-labels and stochastic restore | Continual-setting baseline |
| **On Pitfalls of Test-Time Adaptation (TTAB)** | Zhao, Liu, Alahi, Lin | 2023 | ICML (A\*) | 10 algorithms, many shifts, 2 protocols. Pitfalls: (1) hyperparameter/model selection is very hard from online batch dependency; (2) big sensitivity to pretrained model quality; (3) failure on several shift classes. Concrete failures: BN-adapt 77.8% error, TENT >76% error under severe label shift; TENT/BN-adapt fail under non-stationary shift even with good model selection | **Your motivating critique and your evaluation design source** |

> **RESEARCH GAP STATEMENT (P7).** TTA reliability mechanisms operate at the *sample* level (entropy filtering) or the *loss* level (sharpness-awareness); no widely-adopted method decides at the *batch* level whether to adapt at all, and no study reports the AUROC with which a label-free statistic predicts the **sign** of adaptation benefit across held-out corruption types and adversarial batch compositions.

---

### Landscape for PROJECTS 8, 9, 10 (eliminated)

**P8 / P10 lineage (all Dalca lab, MIT CSAIL):** UniverSeg (Butoi, Gonzalez Ortiz, Ma, Sabuncu, Guttag, Dalca — **ICCV 2023**, pp. 21438–21451) introduced the CrossBlock mechanism and MegaMedical (53 open-access datasets, >22,000 scans, 26 domains, 16 modalities); ScribblePrompt (Wong, Rakic, Guttag, Dalca — **ECCV 2024**) for fast flexible interactive medical segmentation; **Tyche (CVPR 2024)** adds stochastic candidate sets; **MultiverSeg (ICCV 2025)** adds a growing context set to interactive segmentation. Also relevant: SegGPT/Painter, Neuralizer (Czolbe & Dalca), SAM (Kirillov et al., ICCV 2023), MedSAM (Ma et al., Nature Communications 2024).

**Gap statement (P8/P10):** none available to you — the gap these papers identified is the one they closed, and re-closing it requires their training corpus.

**P9 lineage:** RetinexNet (Wei et al., BMVC 2018) introduced the LOL dataset (485 train / 15 test pairs); Zero-DCE (Guo et al., CVPR 2020); SCI (Ma et al., CVPR 2022); URetinex-Net (Wu et al., CVPR 2022); Retinexformer (Cai et al., ICCV 2023); Multinex (CVPR 2026) as above. LOLv2 adds ~789 real and ~1,000 synthetic pairs; LSRW (Hai et al.) provides 5,650 real pairs. Note the widely-acknowledged criticism that LOL-v1 has limited scene diversity and a very small test set, which encourages benchmark overfitting.

**Gap statement (P9):** the remaining gaps are perceptual-quality and no-reference-metric alignment, which are hard to evaluate rigorously and reward engineering effort over scientific insight.

---

## PART 3 — PAPER-TO-PROJECT TRAJECTORIES

Read these as a test of whether each project has a *believable research story* — a chain where each arrow is forced by the previous link, not decorated with buzzwords.

---

**PROJECT 1**
```
FOUNDATIONAL   CLIP (Radford et al., ICML 2021) — image-level contrastive alignment
      ↓
RECENT STRONG  WinCLIP (CVPR'23) → AnomalyCLIP (ICLR'24) → AdaCLIP (ECCV'24) → InCTRL (CVPR'24)
      ↓
LIMITATION     CLIP patch tokens are not spatially supervised; localization lags detection.
               APRIL-GAN already showed extra linear layers help — but never asked WHICH
               inductive bias in the adapter mattered.
      ↓
GAP            Spatial mixing capacity in the adapter is unablated at matched parameter count,
               and gains are never checked against a score-map-smoothing control.
      ↓
PROJECT        Parameter-matched spatial vs. non-spatial adapter, defect-size-stratified.
      ↓
CONTRIBUTION   "Locality in the adapter is/isn't the ingredient — and here is the smoothing
               control that most papers omit."
```
**Verdict:** believable, but every arrow lands in a crowded room. Story strength **6/10**.

---

**PROJECT 2**
```
FOUNDATIONAL   ImageNet-C / CIFAR-C (Hendrycks & Dietterich, ICLR'19) — the measurement instrument
               Fourier perspective (Yin et al., NeurIPS'19) — the explanatory instrument
      ↓
THESIS         Bhojanapalli (ICCV'21), Naseer (NeurIPS'21): ViTs are more robust
      ↓
ANTITHESIS     Bai et al. (NeurIPS'21): those comparisons were unfair; match the recipe and
               CNNs catch up — but self-attention still drives OOD generalization
      ↓
SYNTHESIS      Wang et al. (ICLR'23): patchify + big kernels + fewer norm/act layers ⇒ pure CNNs
               match or beat Transformers. So it wasn't self-attention after all.
      ↓
LIMITATION     Nobody matched on ACHIEVED clean accuracy; nobody reported seed variance;
               nobody reported calibration and accuracy under shift in the same controlled design
               (Minderer et al., NeurIPS'21 did calibration but across uncontrolled model families)
      ↓
GAP            Variance-decompose mCE and ECE across architecture × recipe × seed, at matched ID accuracy
      ↓
PROJECT        Two-tier controlled factorial + Fourier sensitivity as mechanism
      ↓
CONTRIBUTION   A quantitative attribution ("recipe explains X% of the mCE gap, architecture Y%,
               seed Z%") plus the accuracy/calibration dissociation.
```
**Verdict:** every arrow is forced. The field literally reversed itself twice and left the confound un-eliminated. Story strength **9/10**.

---

**PROJECT 3**
```
FOUNDATIONAL   MVTec AD (IJCV'21) → SPADE/PaDiM → PatchCore (CVPR'22, 99.6% AUROC = saturation)
      ↓
RECENT STRONG  VisA (ECCV'22) as a harder benchmark; WinCLIP/AnomalyCLIP as the language branch
      ↓
LIMITATION     Full-shot is solved; few-shot is not. Fusion is fixed-weight and system-level.
               Backbone upgrades and fusion upgrades are never separated.
      ↓
GAP            Backbone family × score family factorial at k≤4
      ↓
PROJECT        2×2 factorial + reliability-gated fusion
      ↓
CONTRIBUTION   "Representation quality dominates fusion design" (or its negation), measured cleanly.
```
**Verdict:** honest and clean; slightly low-ceiling because the answer is plausibly "yes, DINOv2 wins" and everyone half-expects that. Story strength **7/10**.

---

**PROJECT 4**
```
FOUNDATIONAL   Transfer learning; linear probe vs. fine-tune
      ↓
KEY RESULT     Kumar et al. (ICLR'22): FT distorts features, loses OOD; LP-FT fixes it
      ↓
RECENT STRONG  VPT (ECCV'22), LoRA (ICLR'22), BitFit, adapters — all constrain the update
      ↓
LIMITATION     The 2024 unifying PEFT study finds methods perform SIMILARLY once tuned.
               So "which method" is the wrong question.
      ↓
GAP            No label-free predictor of how much OOD accuracy an adaptation will cost
      ↓
PROJECT        Map the label-budget × shift-severity crossover surface; regress OOD gap on feature drift (CKA)
      ↓
CONTRIBUTION   A diagnostic, not a ranking. "Measure your drift, predict your OOD loss."
```
**Verdict:** strong logic, and the pivot away from ranking is exactly right. Story strength **8/10**. Risk: the 2024 unifying study is uncomfortably close.

---

**PROJECT 5**
```
FOUNDATIONAL   Uncertainty sampling; core-set (ICLR'18)
      ↓
KEY RESULT     TypiClust (ICML'22): low vs. high budget need OPPOSITE strategies; cold start is real
      ↓
RECENT STRONG  Gupte et al. (TMLR'24): with DINOv2/OpenCLIP, uncertainty is competitive from
               iteration 2 — the cold start largely evaporates
      ↓
LIMITATION     TCM and similar switch on a hand-set budget, not on a measured signal
      ↓
GAP            Is calibration the observable that tells you when to switch?
      ↓
PROJECT        Calibration-gated acquisition, tested for cross-dataset transfer of the threshold
      ↓
CONTRIBUTION   A budget-free trigger + the backbone-dominates-acquisition result
```
**Verdict:** logically fine, but the TMLR paper already occupies the room and published code. Story strength **6/10**.

---

**PROJECT 6**
```
FOUNDATIONAL   Shortcut learning (Geirhos et al., NMI'20); GroupDRO (ICLR'20) with group labels
      ↓
KEY RESULT     JTT (ICML'21 Oral): error-set upweighting recovers ~73% of the gap with no group labels
      ↓
JTT'S OWN      Error-set GROUP COMPOSITION matters; the SPECIFIC EXAMPLES do not (−0.7% when swapped)
FINDING
      ↓
RIVAL          DFR (ICLR'23 Oral): the representation was fine all along; retrain the last layer,
EXPLANATION    in minutes, and you match SOTA
      ↓
LIMITATION     Identification quality is never measured directly against ground-truth groups
      ↓
GAP            Precision/recall of the inferred error set as a function of persistence evidence
      ↓
PROJECT        Persistence-scored error sets, size-matched, measured against known groups,
               with downstream worst-group accuracy
      ↓
CONTRIBUTION   Either "persistence buys real precision and it converts" or "it buys precision that
               does NOT convert, confirming composition ≫ identity" — both are informative.
```
**Verdict:** the chain is tight and — crucially — **both outcomes are publishable-shaped**. Story strength **8.5/10**.

---

**PROJECT 7**
```
FOUNDATIONAL   Covariate shift; BN statistic adaptation
      ↓
KEY RESULT     TENT (ICLR'21 Spotlight): minimize test entropy, update BN affine params
      ↓
RECENT STRONG  EATA (ICML'22) filters samples; SAR (ICLR'23 Oral) filters + smooths the loss surface
      ↓
CRITIQUE       TTAB (ICML'23): hyperparameter selection is nearly impossible online; catastrophic
               failures documented (>76% error under label shift); several shift classes defeat TTA
      ↓
GAP            All reliability mechanisms are sample- or loss-level; none is a batch-level abstention
      ↓
PROJECT        Batch-level gate; measure sign-prediction AUROC on HELD-OUT corruptions
      ↓
CONTRIBUTION   A harm-conditions map + the finding that standard protocols hide the failure regime
```
**Verdict:** scientifically the most sophisticated story on the list. Story strength **8.5/10** — but see Part 5 and Part 17 for why story strength ≠ grade.

---

**PROJECTS 8, 9, 10** — the trajectory terminates at "PROPOSED PROJECT = the paper you just read." There is no arrow left to draw.

---

## PART 4 — INTRODUCTION-TO-ML COURSE FIT

Scoring 1–10 where 1 = trivially easy for the course and 10 = graduate-seminar level.

| # | Math | ML concepts | CV depth | Coding | Debugging | Research complexity | Prereqs | Scope | Can UG grasp full pipeline? | Can UG defend orally? | **Difficulty** | **Classification** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 7 | 8 | 7 | 8 | 7 | CLIP internals, dense feature extraction, AUPRO | 7 | Partly — CLIP is a black box to most UGs | Hard: "why does your adapter help?" is genuinely difficult | **7.5** | Very risky |
| 2 | 4 | 6 | 7 | 5 | 4 | 6 | Conv/attention basics, augmentation, ECE, FFT | 5 | **Yes, fully** | **Yes** — every choice is a design decision they made | **5.0** | **Appropriate → challenging but reasonable** |
| 3 | 5 | 6 | 7 | 6 | 6 | 6 | kNN/Mahalanobis, coreset, CLIP prompting | 6 | Mostly | Yes, with effort | **6.0** | Challenging but reasonable |
| 4 | 5 | 7 | 5 | 5 | 5 | 6 | PEFT mechanics, CKA, transfer learning | 5 | **Yes** | Yes | **5.5** | **Appropriate → challenging but reasonable** |
| 5 | 6 | 7 | 3 | 5 | 5 | 6 | AL theory, calibration, clustering, heavy stats | 6 | Yes | Yes, but the CV content is thin | **5.5** | Appropriate (weak CV fit) |
| 6 | 5 | 8 | 6 | 6 | 6 | 7 | Group robustness, reweighting, DRO intuition | 6 | Yes | **Yes** — the concepts are crisp | **6.0** | Challenging but reasonable |
| 7 | 6 | 8 | 6 | 7 | **9** | 8 | Online learning dynamics, BN internals, entropy collapse | 7 | Partly | Hard — "why did it collapse?" is not always answerable | **7.5** | Very risky |
| 8 | 7 | 8 | 9 | 9 | 8 | 8 | Medical imaging, in-context architectures, MegaMedical curation | 10 | No | No | **9.0** | Unrealistic |
| 9 | 6 | 5 | 8 | 7 | 7 | 5 | Retinex theory, restoration losses, color spaces | 6 | Mostly | Partly — hard to defend WHY PSNR improved | **6.5** | Challenging but reasonable |
| 10 | 7 | 8 | 9 | 9 | 9 | 8 | Interaction simulation, in-context segmentation | 10 | No | No | **9.0** | Unrealistic |

**The debugging column deserves emphasis.** Project 7's 9/10 is not hyperbole: online adaptation failures are silent. Your model quietly collapses over 200 batches and you cannot tell whether it is a bug, a hyperparameter, or the documented phenomenon. TTAB's own headline pitfall is that model selection is *exceedingly difficult due to online batch dependency*. Two undergraduates debugging that in weeks 4–6 with a report due in week 8 is a real hazard.

Project 2's debugging score of 4 is its quiet superpower. A supervised classifier that trains badly tells you immediately, in the loss curve.

---

## PART 5 — COMPUTATIONAL FEASIBILITY

Assumptions: single A100 40/80GB when available; assume you get roughly **60–100 GPU-hours total across 8 weeks**, in blocks, not continuously. Mixed precision (AMP) available everywhere. No multi-GPU.

| # | Dataset scale | Model(s) | #models trained | GPU mem | Train load | Experiment count | Est. total GPU-h | **Rating** | Does the A100 change feasibility? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | MVTec 5.3k + VisA 10.8k imgs | CLIP ViT-B/16 frozen + ≤1M-param adapter | ~12 adapter trainings | 8–16 GB | Light (frozen backbone) | ~60 eval configs | 25–40 | **YELLOW** | Moderately — feature caching matters more than the GPU |
| 2 | CIFAR-100 (60k, 32×32) or Tiny-ImageNet (110k, 64×64) + corruption sets | ResNet-18 (11M), ViT-Tiny (5M) / Swin-Tiny (28M) | **18–24 from scratch** | 6–12 GB | Moderate; each run 30–90 min with AMP | ~24 train runs × 95 corruption evals | **25–45** | **GREEN** | Yes, materially — turns a 3-week job into a 4-day job. But a single RTX 3090/4090 or free Colab T4 still completes the MINIMUM version. |
| 3 | MVTec 5.3k + VisA 10.8k | WRN-50, DINOv2 ViT-B, CLIP ViT-B — **all frozen** | **0 trained** | 8–16 GB | **Near-zero — feature extraction only** | ~200 cheap eval configs | **8–15** | **GREEN** | Barely — this project is almost GPU-free after caching |
| 4 | Source ~10–50k imgs + 2–3 shift sets | ViT-B/16 frozen + PEFT heads | ~60 short adaptations | 12–24 GB | Light–moderate | 5 methods × 4 budgets × 3 seeds × 4 eval sets | 20–35 | **GREEN** | Helpful, not required |
| 5 | CIFAR-100 / a VTAB-style set, **embeddings cached** | 3 frozen backbones + linear heads | 0 backbone trainings | <8 GB after caching | **Trivial** — linear heads on cached vectors | 6 acq × 3 backbones × 8 rounds × 5 seeds = 720 probe fits | **5–10** | **GREEN** | No — runs on a laptop after one caching pass |
| 6 | Waterbirds (~11.8k) + CelebA (~200k, subsample) | ResNet-50 pretrained, fine-tuned | ~20–30 short runs | 12–24 GB | Moderate (ResNet-50 @224) | 5 methods × 2 datasets × 3–5 seeds | 20–35 | **GREEN** | Yes, comfortably; CelebA is the only heavy part and can be subsampled |
| 7 | CIFAR-10-C / CIFAR-100-C (950k corrupted imgs total across 19×5) | WRN-28-10 or ResNet-26 source + 4 TTA methods | 1–2 source models | 8–16 GB | Light per run, but **hundreds of adaptation streams** | 4 methods × 15 corruptions × 5 severities × 4 batch configs × 3 seeds ≈ 3,600 streams | 30–50 | **YELLOW** | Yes — but compute is not the risk; convergence pathology is |
| 8 | MegaMedical: 53 datasets, >22k scans | UniverSeg-class in-context net | 1 huge | 40+ GB | **Weeks of A100 time** | — | 300+ | **RED** | No. Out of reach. |
| 9 | LOL-v1 (485/15), LOL-v2 (~789 real, ~1000 syn) | 45K–2M param enhancement nets | ~10 | 8–16 GB | Moderate, long schedules (BasicSR configs) | ~30 | 25–45 | **YELLOW** | Somewhat; the risk is metric-chasing on a 15-image test set |
| 10 | MegaMedical + simulated interaction loops | MultiverSeg-class net | 1 huge | 40+ GB | **Weeks** | — | 300+ | **RED** | No. |

### The A100 discipline rule
The A100 should buy you **more seeds and more ablation cells**, not a bigger model. Concretely, for the recommended project, the A100 converts "3 seeds is too expensive, we'll report 1" into "5 seeds with 95% CIs" — and *that* is the difference between an 82% and a 93% project. A bigger backbone would buy you nothing your professor cares about.

**Explicit fallback plan if A100 access never materializes for Project 2:** drop from Tiny-ImageNet to CIFAR-100 (32×32), from Swin-Tiny to ViT-Tiny, from 5 seeds to 3, and from 24 to 12 training runs. That version fits on a **single free Colab T4 in ~20 hours of wall-clock**, and still answers the primary hypothesis. No other project on this list has a fallback that clean.

---

## PART 6 — DATASET ANALYSIS

I have deliberately kept this to one primary and one secondary per project. Under an 8-week deadline, a third dataset is a liability, not a strength.

### Project 1 & 3 (industrial AD)
**PRIMARY — MVTec AD.** Task: unsupervised anomaly detection + pixel-level localization. Scale: 15 categories, ~5,354 images (3,629 nominal train / 1,725 test), resolutions 700×700–1024×1024. Availability: free for **non-commercial research**, registration required via MVTec's site — allow 1–3 days. Structure: train = nominal only; test = nominal + defective with pixel masks; **no official validation split** (you must carve one, and say so). Fit: the canonical cold-start benchmark. Difficulty: **saturated at full-shot (PatchCore 99.6%)** — only use it in the k≤4 regime or you will have no headroom.
**SECONDARY — VisA.** 10,821 images (9,621 normal / 1,200 anomalous), 12 objects across 3 domains (complex-structure PCBs, multi-instance, single-instance), 78 anomaly types, image + pixel labels, captured at 4000×6000. Available on AWS Open Data. Harder than MVTec: smaller defects, noisy backgrounds, variable object pose. Fit: the generalization test that proves you didn't overfit MVTec.

### Project 2 (architectural robustness) — **the cleanest dataset story on the list**
**PRIMARY — CIFAR-100 + CIFAR-100-C.** Task: 100-way classification, then corruption robustness. Scale: 50k train / 10k test at 32×32; CIFAR-100-C applies **15 corruptions × 5 severities** to the test set (19 corruption types are distributed, 15 are the standard evaluation set; the extras — speckle noise, Gaussian blur, spatter, saturate — are designated *validation* corruptions and must **not** be used for tuning if you evaluate on the main 15). Availability: instant, unrestricted, Zenodo-hosted. Structure: carve 5k from train as validation; never touch the corrupted set until final eval. Supports the RQ: **perfectly** — it is the instrument the field uses.
**SECONDARY — Tiny-ImageNet + Tiny-ImageNet-C** (200 classes, 64×64) as a scale-generalization check, *or* a small ImageNet-C subset for the pretrained-model tier.
**Critical methodological note you must state in the report:** Hendrycks & Dietterich explicitly instruct that networks be trained on clean data and **not** on the corruption images. Your professor may test you on this. Training on ImageNet-C corruptions and reporting ImageNet-C accuracy is the classic leakage failure in this literature.

### Project 4 (PEFT under shift)
**PRIMARY — DomainNet (subset) or Office-Home.** Office-Home: 4 domains (Art, Clipart, Product, Real-World), 65 classes, ~15,500 images — the right size for 8 weeks. Availability: free, direct download. Structure: train on one domain at n∈{1,5,25,all} per class, test on the other three. Supports the RQ directly: label budget and shift severity are both natively controllable.
**SECONDARY — CIFAR-10 → STL-10** or **ImageNet→ImageNet-R/Sketch** for a second shift type (Kumar et al. used exactly these families).

### Project 5 (active learning)
**PRIMARY — CIFAR-100** (embeddings cached once from DINOv2/CLIP). **SECONDARY — a biomedical set** such as a public histology or dermoscopy classification dataset, because the TMLR paper flagged out-of-domain biomedical images as understudied in AL — that's where your marginal value is. Check licensing before committing.

### Project 6 (shortcut learning) — **second-cleanest dataset story**
**PRIMARY — Waterbirds.** Task: binary landbird/waterbird classification where background (land/water) is the spurious attribute. Scale: ~11,788 images, constructed from CUB birds composited onto Places backgrounds. Availability: free; standard splits distributed with the GroupDRO codebase. Structure: **train/val/test all carry ground-truth group labels** — this is exactly what makes your precision/recall measurement possible. Note the label imbalance the JTT authors documented (waterbird label appears in ~23% of training examples), which you must control for when size-matching error sets.
**SECONDARY — CelebA (blond/non-blond with gender as spurious attribute).** ~200k images; subsample to ~50k for tractability and say so. Blond appears in ~15% of training examples. Second dataset confirms your persistence finding isn't Waterbirds-specific.
**Licensing caution:** CelebA is for **non-commercial research only**; state this in your report.

### Project 7 (TTA)
**PRIMARY — CIFAR-10-C.** **SECONDARY — CIFAR-100-C.** Both are the field-standard TTA testbeds and are what TTAB itself uses. Avoid ImageNet-C for the main study — the adaptation streams get too long.

### Projects 8/9/10
P8/P10 need MegaMedical (53 datasets, >22k scans) — assembling and licensing that alone exceeds 8 weeks. P9 needs LOL-v1/v2 + no-reference sets (DICM, MEF, NPE, LIME), all freely available, but with the 15-image test set caveat.

---

## PART 7 — BASELINE LADDERS

The rule you set — never jump straight to the sophisticated model — is exactly right, and it is also the single easiest way to signal research maturity. Here is the ladder for each.

| # | Simple baseline | Strong standard baseline | Strong modern baseline | Proposed / advanced |
|---|---|---|---|---|
| 1 | Raw CLIP patch–text cosine similarity, no adapter | k-shot nearest-neighbour memory bank on CLIP patch features | WinCLIP-style window ensemble; APRIL-GAN-style linear adapter | Parameter-matched **spatial** adapter + fused memory score |
| **2** | **ResNet-18 with basic aug (flip+crop), evaluated on clean + corrupted** | **ResNet-18 with the full DeiT recipe** (the fair-comparison move) | **ViT-Tiny/Swin-Tiny with the identical recipe, matched on clean accuracy** | **Full architecture × recipe factorial with variance decomposition + Fourier profiles + ECE** |
| 3 | Global-average-pooled ImageNet feature + kNN distance | PaDiM (patch Gaussian + Mahalanobis) | PatchCore with coreset subsampling; WinCLIP zero-shot | DINOv2-backbone PatchCore + reliability-gated hybrid fusion |
| 4 | Linear probe on frozen features | Full fine-tuning | LoRA / VPT / BitFit / adapters, each properly tuned; LP-FT | Drift-regularized or drift-monitored adaptation with the predictive diagnostic |
| 5 | Random sampling (the baseline everyone forgets and that often wins) | Entropy / margin uncertainty sampling | TypiClust, ProbCover, BADGE; TCM switch heuristic | Calibration-gated switch |
| **6** | **ERM (no intervention)** | **JTT (single-run error set)** | **DFR (last-layer retraining) + GroupDRO oracle upper bound** | **Persistence-scored error set + size-matched reweighting** |
| 7 | Source model, no adaptation | BN-statistic adaptation only | TENT; EATA or SAR as the modern reliability-aware method | Batch-level reliability gate on top of TENT/EATA |
| 9 | Histogram equalization / gamma correction | Zero-DCE (unsupervised) | Retinexformer or a lightweight SOTA | Reduced-parameter multi-prior variant |

**Why each rung is scientifically necessary — using Project 2 as the worked example:**
- **Rung 1** establishes that your training pipeline is correct and gives the "how bad is naive" reference. Without it you cannot claim any gain is meaningful.
- **Rung 2** is the *critical* one and the one most students skip. Giving the CNN the ViT's recipe is precisely the manipulation Bai et al. showed changes the conclusion. If you omit it, your entire result is a confound.
- **Rung 3** is the architecture arm under the identical recipe, matched on clean accuracy. Matching on accuracy rather than parameters is your protocol contribution.
- **Rung 4** is not a "better model" — it is the *analysis* (factorial decomposition + ECE + Fourier). In a study like this, the contribution lives in rung 4's analysis, not in a new architecture. **Say this explicitly in your report**; it preempts the "where's your method?" question.

For **Project 6**, note that GroupDRO is an *oracle upper bound*, not a competitor — it uses training group labels you are pretending not to have. Framing it that way in your results table ("oracle: 89.9%") is a maturity signal.

---

## PART 8 — EXPERIMENT DESIGN

I give the full design for the top four (2, 6, 4, 3) and a compressed design for the rest, because designing 10 full experiment suites would encourage you to over-scope.

---

### PROJECT 2 — full experiment plan

**Core experiments (non-negotiable).**
- E1. Clean accuracy for every (architecture, recipe, seed) cell — establishes the accuracy-matching.
- E2. Corruption robustness: accuracy on all 15 corruptions × 5 severities → per-corruption error, mCE, and relative mCE.
- E3. Calibration: ECE (15-bin), before and after temperature scaling fitted **on clean validation only**, evaluated at every severity.

**Comparison experiments.**
- ResNet-18 vs. ViT-Tiny vs. Swin-Tiny, all at matched clean accuracy (±0.5%) via early-stopping selection or width adjustment. Report the matching procedure and the residual accuracy gap honestly.
- Tier-A extension: 8–12 pretrained `timm` models spanning ResNet/ConvNeXt/DeiT/Swin evaluated on an ImageNet-C subset, to show your CIFAR conclusion is not scale-specific.

**Ablation studies (each factor removed *independently*).**
- A1. Augmentation: none → flip+crop → RandAugment → +Mixup/CutMix → +AugMix.
- A2. Normalization: BatchNorm vs. GroupNorm vs. LayerNorm in the CNN (this directly probes Wang et al.'s "reduce normalization layers" claim).
- A3. Patchification: standard ResNet stem vs. a patchify stem (their claim #1) — a ~10-line change, highest-value ablation on the list.
- A4. Kernel size in the CNN: 3×3 vs. 7×7 vs. 11×11 depthwise (their claim #2).
- A5. Training length: 100 vs. 200 epochs (guards against "the ViT was just undertrained").

**Robustness experiments.** Full corruption grid, plus severity-response curves (accuracy vs. severity per architecture) — these curves make an excellent figure because the *slopes*, not the intercepts, carry the argument.

**Generalization experiments.**
- Held-out corruption families: tune nothing on the 4 designated validation corruptions; report on the 15 test corruptions.
- Cross-dataset: CIFAR-100-C → Tiny-ImageNet-C, same protocol.
- Natural shift: CIFAR-10 → STL-10 (shared classes) as a non-synthetic check.

**Error analysis.** Confusion-matrix deltas clean→corrupted per architecture; which class pairs collapse first; whether ViT and CNN fail on the *same* images (compute error-set overlap / Jaccard — a genuinely interesting, cheap analysis nobody expects from an undergrad).

**Qualitative analysis.** Grad-CAM (CNN) vs. attention rollout (ViT) on identical images at severity 0 and 5; Fourier sensitivity heatmaps per architecture; reliability diagrams at severity 0, 3, 5.

**Statistical analysis.** **3 seeds minimum, 5 preferred.** Report mean ± std for every cell. Two-way ANOVA on mCE with factors {architecture, recipe} and the interaction term — this single table is what elevates the project from "study" to "controlled experiment." Bootstrap 95% CIs on mCE differences. State explicitly that a 1-point mCE difference within ±2 std is **not** a result.

**8-week fit:** ~24 training runs at ≤90 min each = ~30 GPU-hours, plus cheap evaluation. Comfortable.

---

### PROJECT 6 — full experiment plan

**Core.** Worst-group accuracy and average accuracy on Waterbirds and CelebA for: ERM, JTT, DFR, GroupDRO (oracle), and your persistence variants.
**Comparison.** Error-set constructions at **matched set size**: (a) single-run final-epoch errors (JTT), (b) intersection across s seeds, (c) persistence score = fraction of seeds misclassifying, thresholded, (d) epoch-persistence within one run, (e) random subset of the same size (the null control — **essential**).
**Ablations.** Number of seeds s ∈ {1,2,3,5}; upweight factor λ; identification-model training length T; error-set size.
**Robustness.** Vary the spurious correlation strength by resampling the training set; check whether persistence's advantage grows with correlation strength.
**Generalization.** Fit λ and threshold on Waterbirds, apply unchanged to CelebA.
**Error analysis.** **The headline measurement:** precision and recall of each error set against ground-truth minority-group membership, plotted against downstream worst-group accuracy. If the correlation is weak, you have empirically confirmed and extended the JTT authors' composition-over-identity finding.
**Qualitative.** Grid of images that are persistently wrong vs. sporadically wrong; background-swap visualizations; Grad-CAM before/after.
**Statistical.** 5 seeds for the final numbers; worst-group accuracy is high-variance on Waterbirds and reporting a single run is a common and fatal error.

---

### PROJECT 4 — compressed plan
Core: ID and OOD accuracy for 5 adaptation methods × 4 budgets × 3 seeds. Ablations: LoRA rank, VPT prompt length, adapter bottleneck — all at matched trainable-parameter count. Robustness: 3 target domains at increasing distance. Generalization: transfer the drift→OOD-gap regression fitted on domain A to domains B and C. Error analysis: per-class OOD degradation vs. class frequency in the adaptation set. Statistical: report the crossover budget with a bootstrap CI.

### PROJECT 3 — compressed plan
Core: image-AUROC, pixel-AUROC, AUPRO at k∈{1,2,4,8} for each scorer and each fusion rule. Ablations: backbone (WRN-50 / DINOv2 / CLIP) × score (memory / text / both) 2×3; coreset ratio; feature layer; score normalization scheme. Robustness: transfer MVTec→VisA with no retuning. Error analysis: per-defect-type breakdown; which categories does the text score rescue. Statistical: 5 random k-shot draws per category, report mean ± std — **essential**, because k=1 results swing wildly with which reference image you draw.

### PROJECTS 1, 5, 7 — compressed plans
**P1:** spatial vs. non-spatial adapter at matched params; defect-size strata (small/medium/large by mask area); the Gaussian-smoothing control; 3 seeds; MVTec→VisA transfer.
**P5:** 6 acquisitions × 3 backbones × 8 rounds × 5 seeds; ECE tracked per round; gate threshold fitted on dataset A, transferred to B; random-sampling baseline shown on every plot.
**P7:** 4 methods × 15 corruptions × 5 severities × {batch size 16/64/200} × {uniform/skewed labels} × {episodic/continual} × 3 seeds — **note this is already ~5,000 streams and you must cut it.** Recommended cut: severities {3,5} only, batch sizes {16,200} only, → ~1,300 streams.

---

## PART 9 — EVALUATION METRICS

Every metric below is justified by *what question it answers*, not by convention.

### Project 2 (recommended)
| Metric | Why it matters for THIS research question |
|---|---|
| **Clean top-1 accuracy** | It is the *matching variable*. Without it the OOD comparison is meaningless — you cannot attribute an OOD gap to architecture if the models differ in-distribution. |
| **mCE (mean corruption error)** | The field-standard aggregate, normalized against an AlexNet reference per corruption, which prevents easy corruptions from dominating the average. Using it makes your numbers comparable to the literature. |
| **Relative mCE (corruption error minus clean error)** | Separates "this model is worse everywhere" from "this model degrades faster." Your hypothesis is about *degradation*, so this is arguably your primary metric. |
| **Per-corruption error** | Because the Fourier argument predicts *which* corruptions each architecture fails on. An aggregate would hide the mechanism. |
| **ECE (15-bin) at each severity** | Directly tests H2. A model can be equally accurate and far more overconfident — that difference is invisible to accuracy. |
| **ECE after temperature scaling (fitted on clean val)** | Distinguishes a fixable confidence bias from a genuine ranking failure. Minderer et al. show the ViT/Mixer advantage persists after temperature scaling, so this is the right control. |
| **Seed standard deviation** | Not decoration. It is the denominator against which every claimed gap must be judged. |
| Params / FLOPs / inference latency | Secondary. Include a small table so the reader can judge whether a robustness gain is worth its cost. |

**Deliberately excluded:** AUROC (not a classification-accuracy question), IoU/Dice (no segmentation), PSNR/SSIM (no restoration). Forcing those in would be a red flag.

### Project 6
Worst-group accuracy (**primary** — the entire point), average accuracy (to show the trade-off, which JTT quantified at ~3.7%), gap-to-oracle as a percentage of the ERM→GroupDRO gap (JTT's own framing at ~73%, so it makes you directly comparable), error-set precision/recall vs. ground-truth groups (**your novel measurement**), and total GPU-hours per method (because DFR's headline claim is cost, and you must engage with it).

### Project 3 / Project 1
Image-level AUROC (detection), **AUPRO** (localization — the AD-community standard because plain pixel AUROC is dominated by the vast normal background), pixel AP / F1-max (better under the extreme imbalance of defect pixels), per-category and per-defect-type breakdown, inference latency and memory-bank size (relevant to the industrial framing).

### Project 4
ID accuracy, OOD accuracy, **ID−OOD gap** (the quantity under study), trainable parameter count (the "efficient" in PEFT), CKA/feature-drift (your predictor), and adaptation wall-clock.

### Project 5
Accuracy vs. labeling budget curves, **AUBC** (area under the budget curve, to summarize a curve as a number), ECE per round (your gate signal), and **accuracy improvement over random** — because random sampling is the baseline that beats most AL methods at low budget, and omitting it is the most common dishonesty in the AL literature.

### Project 7
Online error rate, **worst-case error across corruptions** (the harm question), sign-prediction AUROC of the gate, adaptation cost in forward/backward passes per sample, and collapse rate (fraction of streams ending above source-only error).

---

## PART 10 — NOVELTY ANALYSIS

I am being deliberately conservative here. For each project I separate what exists, what you'd reproduce, what you'd change, what could count as contribution, and — importantly — **what would not count**.

---

### PROJECT 1
**A. Existing work.** CLIP-based zero-/few-shot AD via window ensembles (WinCLIP), object-agnostic learned prompts (AnomalyCLIP), hybrid learnable prompts (AdaCLIP), added linear projection layers + memory banks (APRIL-GAN), in-context residual learning (InCTRL).
**B. Reproduction.** WinCLIP-style scoring; a k-shot memory bank; MVTec/VisA evaluation protocol.
**C. Modification.** Replace the per-token projection with a spatially-mixing module of identical parameter count.
**D. Possible contribution.** (i) The parameter-matched spatial-vs-nonspatial ablation itself; (ii) defect-size-stratified reporting; (iii) the Gaussian-smoothing null control quantifying how much apparent localization gain is just blur.
**E. Non-novel changes.** Swapping ViT-B/16 for ViT-L/14; adding more prompt templates; tuning the fusion weight; changing the feature layer index; using a larger memory bank.
**Three realistic novelty directions:** (1) spatial-inductive-bias ablation in the adapter; (2) a smoothing-controlled localization protocol; (3) defect-scale-conditional evaluation.

---

### PROJECT 2 (recommended)
**A. Existing work.** ImageNet-C/CIFAR-C benchmarks; Fourier sensitivity analysis; the three-way thesis/antithesis/synthesis on CNN-vs-ViT robustness; the large-scale calibration survey.
**B. Reproduction.** Bai et al.'s unified-recipe protocol at small scale; Hendrycks & Dietterich's mCE; Yin et al.'s Fourier heatmaps; Minderer et al.'s ECE-under-shift measurement.
**C. Modification.** (i) Match on **achieved clean accuracy** rather than parameter count or nominal scale; (ii) cross architecture × recipe as explicit factors with replicate seeds instead of comparing two fixed configurations; (iii) measure accuracy and calibration in the same controlled design.
**D. Possible contribution.**
1. **A variance decomposition of mCE** attributing the robustness gap to architecture, recipe, their interaction, and seed noise. To my knowledge this specific decomposition, with an ANOVA-style table, is not reported in the CNN-vs-ViT literature — the frontier papers report point comparisons.
2. **The accuracy/calibration dissociation test.** Whether recipe-matching that eliminates the accuracy gap also eliminates the calibration gap. Minderer et al. give the prior; nobody has tested it under a controlled recipe manipulation.
3. **A matched-accuracy fair-comparison protocol** as a reusable evaluation contribution.
4. **Error-set overlap analysis** — do CNNs and ViTs fail on the same corrupted images, or on disjoint ones? Cheap, visual, and rarely reported.
**E. Non-novel changes.** Training longer; using a bigger ViT; adding a new corruption type; reporting on a different dataset with no new control; changing the optimizer or learning rate; "we also tried ConvNeXt."
**Three realistic novelty directions:** (1) matched-accuracy protocol + variance decomposition; (2) calibration-under-shift as a second, dissociable axis; (3) Fourier-profile-based mechanistic explanation linking corruption-family failures to architecture, at a scale two students can actually run.

**Honest calibration of this novelty:** it is **methodological and analytical**, not architectural. That is fine — arguably better for an Intro ML course, because you must *understand* experimental design to produce it, and you cannot fake it. But you must not oversell it as a new method. Frame it as: *"we do not propose a new architecture; we propose a protocol and we report what it reveals."* Professors respect that framing far more than a fake method.

---

### PROJECT 3
**A. Existing.** PaDiM, PatchCore, SPADE; WinCLIP/AnomalyCLIP; DINOv2 features; various fusion schemes inside individual papers.
**B. Reproduction.** PatchCore (or `anomalib`'s implementation) and a WinCLIP-style zero-shot scorer.
**C. Modification.** Backbone × score-family factorial; reliability-gated rather than fixed-weight fusion.
**D. Contribution.** The orthogonal attribution of few-shot AD gains to representation vs. fusion; a per-image reliability signal for fusion weighting; MVTec→VisA transfer with no retuning.
**E. Non-novel.** Fixed-weight score averaging; changing the coreset ratio; adding a third backbone with no design rationale; reporting MVTec full-shot AUROC (saturated).
**Three directions:** (1) 2×2 backbone/score factorial; (2) reliability-gated fusion with a coverage statistic; (3) a few-shot-variance protocol (k-shot draws as a reported source of variance — currently under-reported).

---

### PROJECT 4
**A. Existing.** LP vs. FT vs. LP-FT under shift; LoRA/VPT/BitFit/adapters; the 2024 unifying PEFT study finding near-parity after proper tuning.
**B. Reproduction.** The LP-vs-FT OOD crossover; standard PEFT implementations.
**C. Modification.** Add the label-budget dimension as a continuous axis; add a feature-drift measurement to every run.
**D. Contribution.** The drift→OOD-degradation regression as a **label-free diagnostic** that transfers across PEFT families; the crossover surface in (n, shift severity).
**E. Non-novel.** Reporting that LoRA beats VPT on some dataset; tuning ranks; adding another PEFT method to the table.
**Three directions:** (1) drift-as-predictor diagnostic; (2) budget×severity crossover surface; (3) drift-regularized adaptation (a genuine but optional method extension).

---

### PROJECT 5
**A. Existing.** TypiClust, ProbCover, BADGE, coreset, BALD; the TMLR foundation-model AL study; the TCM budget-switch heuristic.
**B. Reproduction.** The TMLR finding that uncertainty becomes competitive early with strong features.
**C. Modification.** Replace the budget-based switch trigger with a calibration-based one.
**D. Contribution.** A measured, budget-free switching signal validated for cross-dataset transfer; the backbone-dominates-acquisition variance comparison.
**E. Non-novel.** Trying a new acquisition score; using a bigger backbone; running on one more dataset.
**Three directions:** (1) calibration-gated switching; (2) variance attribution across backbone vs. acquisition; (3) an AL protocol that reports random-baseline-relative gains with CIs (an evaluation-protocol contribution — AL badly needs one).

---

### PROJECT 6
**A. Existing.** GroupDRO (with labels); LfF and JTT (identify-then-reweight); DFR (last-layer retraining); LaBonte et al. (fewer annotations for last-layer retraining).
**B. Reproduction.** ERM, JTT, DFR on Waterbirds and CelebA.
**C. Modification.** Change the *identification signal* from single-run errors to persistence-across-seeds/epochs, with strict set-size matching and a random-subset null.
**D. Contribution.** (i) The precision/recall-vs-worst-group-accuracy measurement — i.e., quantifying whether identification quality is the binding constraint at all; (ii) the persistence-vs-compute curve (how many seeds buy how much precision); (iii) a direct empirical test, at controlled set size, of the JTT authors' own composition-over-identity observation.
**E. Non-novel.** Changing the upweight factor λ; training the identification model for a different number of epochs; using a different backbone; simply running JTT with 3 seeds and averaging results.
**Three directions:** (1) persistence-based identification; (2) identification-quality-vs-outcome measurement; (3) a compute-normalized comparison of identify-then-reweight vs. last-layer-retraining (nobody reports these at equal GPU-hours, and DFR's central claim is about cost).

---

### PROJECT 7
**A. Existing.** TENT; EATA's entropy filtering + Fisher regularizer; SAR's filtering + sharpness-aware minimization; CoTTA; TTAB's critique; recent reliability-gated continual TTA work.
**B. Reproduction.** TENT and BN-adapt on CIFAR-C; TTAB's adversarial evaluation conditions.
**C. Modification.** Move the reliability decision from sample level to batch level; add abstention.
**D. Contribution.** A harm-conditions map; sign-prediction AUROC on **held-out** corruption types; the demonstration that the standard protocol conceals the failure regime.
**E. Non-novel.** A new entropy threshold; a different learning rate; adapting more layers; combining two existing losses.
**Three directions:** (1) batch-level abstention gate; (2) sign-prediction as a new evaluation quantity for TTA; (3) an adversarial-stream protocol (small batches, skewed labels, mild shift).

---

### PROJECTS 8, 9, 10
**Novelty available to you: essentially zero for 8 and 10** (you cannot train the model, so you cannot modify it meaningfully). For **9**, novelty would be architectural micro-design on a 15-image test set — the highest effort-to-insight ratio on this list.

---

## PART 11 — HYPOTHESIS STRENGTH

| # | Primary hypothesis | Measurable? | Falsifiable? | Directly tested? | Could it be wrong? | Is a negative result interesting? |
|---|---|---|---|---|---|---|
| 1 | Spatial adapter > matched non-spatial adapter on AUPRO at k=4 | Yes | Yes | Yes | Yes — locality may already be captured by window ensembling | **Moderately.** "Locality isn't the bottleneck" is useful but reads as a null on a narrow question |
| **2** | **Recipe-matching + accuracy-matching collapses most of the CNN–ViT mCE gap; the calibration gap survives** | **Yes, quantitatively (variance shares)** | **Yes** | **Yes — the factorial IS the test** | **Yes.** Wang et al.'s ICLR'23 result makes either outcome plausible | **Strongly.** If the architecture effect survives matching, you've *confirmed* the architectural hypothesis under a stricter control than the literature applied — that is a positive result either way. **This hypothesis cannot produce an uninteresting outcome.** |
| 3 | Reliability-gated fusion > fixed fusion at k=1, decaying with k | Yes | Yes | Yes | Yes | Moderate. H2 (backbone dominates) is the interesting one and *is* the likely outcome |
| 4 | ID/OOD ranking inverts with budget × severity; drift predicts the OOD gap | Yes | Yes | Yes | Yes — drift may not correlate across families | **Yes.** "Drift does not predict OOD loss" refutes a natural extension of feature-distortion theory, which is a real finding |
| 5 | Calibration threshold predicts the diversity→uncertainty switch point | Yes | Yes | Yes | Yes | Moderate. A failed gate is a weak paper unless H2 (backbone dominance) carries it |
| **6** | **Persistence-filtered error sets have higher group precision AND convert to higher worst-group accuracy** | **Yes** | **Yes** | **Yes** | **Yes — JTT's own ablation predicts it may not convert** | **Strongly.** A precision gain that does NOT convert is a clean confirmation-and-extension of composition-over-identity, and is arguably the more interesting outcome |
| 7 | A batch-level gate predicts the sign of TTA benefit and eliminates worst-case harm | Yes | Yes | Yes | Yes | **Yes**, but only if you get clean measurements — and clean measurements are the risky part |

**The two hypotheses where a negative result is genuinely as good as a positive one are Project 2's and Project 6's.** That property is worth an enormous amount under an 8-week deadline, because it decouples your grade from whether nature cooperates.

Project 2 deserves a specific note. Its hypothesis is *symmetric*: the study is designed to **measure a decomposition**, not to win a comparison. There is no outcome in which you have "no result." You will always have a variance table. Compare that to Project 1, where "the adapter didn't help" leaves you with a thin story.

---

## PART 12 — PROFESSOR APPEAL

Speaking now as the CV professor grading this.

### What makes me interested vs. suspicious, per project

**Project 1.** *Interested by:* dense CLIP features, localization, industrial relevance. *Suspicious that:* you are gluing modules onto a foundation model and hoping. I will ask you to explain what CLIP's patch tokens actually represent and why the text encoder should be expected to describe a scratch. If you cannot, this collapses.

**Project 2.** *Interested by:* you identified a live contradiction in A\*-venue literature and designed a controlled experiment to adjudicate it. *This is what research is.* You are demonstrating that you understand **confounding**, which is the single most important idea in empirical ML and the one undergraduates most reliably lack. *Suspicious that:* it becomes a benchmark table. The factorial, the seeds, the ANOVA, and the Fourier analysis are what prevent that — and each one is a specific thing I can point to in your report.
*What makes me think "this student genuinely understands the science":* the moment you say "we matched on achieved clean accuracy rather than parameter count, because a model that is 2 points worse in-distribution isn't comparable out-of-distribution." That one sentence tells me you thought like an experimentalist.
*What makes me think "flashy implementation":* if you report a single seed, or if you say "ViTs are more robust" without a variance estimate.

**Project 3.** *Interested by:* clean factorial thinking, real industrial relevance, beautiful anomaly maps. *Suspicious that:* MVTec is saturated and you're chasing noise. **Report the few-shot regime only, and say why.** If you show me PatchCore's 99.6% full-shot number and then claim a 0.2% improvement, I will not be impressed.

**Project 4.** *Interested by:* it addresses the decision practitioners actually make, and the feature-distortion theory is elegant. *Suspicious that:* it's a hyperparameter sweep with a nice title. The CKA-drift predictor is what saves it — it converts a sweep into a hypothesis test.

**Project 5.** *Interested by:* calibration as a control signal is a nice idea. *Suspicious that:* there is almost no computer vision in it. I am a CV professor; a project where images are only ever frozen embedding vectors gives me nothing to engage with visually. Also I will find the TMLR paper in ten seconds.

**Project 6.** *Interested by:* shortcut learning is central to why vision systems fail in deployment, and Waterbirds visualizations are compelling. *Suspicious that:* worst-group accuracy on Waterbirds is notoriously noisy and you will report one seed. Also: DFR exists and is cheaper than everything; if you don't engage with it, your framing is out of date by three years.

**Project 7.** *Interested by:* "when does the method hurt" is a more mature question than "does the method help." *Suspicious that:* you will spend six weeks debugging non-convergence and hand me three tables. The TTAB paper's own headline is that this is hard.

**Project 9.** *Interested by:* the visuals are the best on the list. *Suspicious that:* PSNR on a 15-image test set is not evidence, and I will ask you to defend that. If your answer is "we also did no-reference metrics on DICM/LIME," that's better; if not, this is a demo, not a study.

**Projects 8, 10.** *Suspicious immediately:* I will recognize the titles. Full stop.

### The generalizable signal
What separates "genuinely understands the science" from "flashy implementation" across every project is the same three things:
1. **Do you report variance, and do you interpret gaps relative to it?**
2. **Did you run the control that could have falsified your story?** (the smoothing control in P1; the random-subset error set in P6; the recipe-matched CNN in P2; the random-sampling baseline in P5)
3. **Can you explain *why* the result happened, not just that it happened?** (Fourier profiles; CKA drift; error-set precision; per-defect-type breakdown)

---

## PART 13 — ORAL DEFENSE SIMULATION

### Project 2 — the questions I would actually ask
1. You matched clean accuracy between the CNN and the ViT. How exactly? Early stopping? Width? And what residual accuracy gap remains — is it inside your seed noise?
2. Bai et al. concluded the OOD advantage comes from self-attention itself. Wang et al. concluded patchification and kernel size suffice. Your factorial — which of them does it support, and could your design distinguish them at all?
3. Your ViT-Tiny trained from scratch on CIFAR-100 needs heavy augmentation to work at all. Doesn't that make "architecture" and "recipe" inseparable in your design? How do you answer that?
4. Why mCE and not plain average corruption accuracy? What does the AlexNet normalization buy you, and could it distort your conclusion?
5. You fit temperature scaling on clean validation data. Why not on corrupted data — and what would change if you did?
6. What in your protocol prevents leakage from the corruption benchmark into training?
7. Your ANOVA gives an interaction term. Interpret it physically — what does a large architecture×recipe interaction *mean* about vision models?
8. If your result had come out the other way — architecture explaining most of the variance — what would you have concluded, and would you have believed it?
9. Your Fourier heatmaps show the ViT is less sensitive to high-frequency perturbations. Is that a cause of its corruption robustness or a consequence of the same training choices?
10. Two undergraduates with 30 GPU-hours are contradicting NeurIPS papers. What is the honest scope of your claim?

**Oral-defense difficulty: 6/10.** Hard enough to be respectable; every question is answerable by students who did the work, because every question is about *their own design decisions*. That is the ideal profile.

### Project 6 — questions
1. Define persistence precisely. Persistence across seeds and persistence across epochs measure different things — which are you claiming and why?
2. You matched error-set sizes. Why is that essential, and what happens without it?
3. JTT's authors found that swapping error-set examples within the same group cost only 0.7%. If they're right, why should persistence help at all?
4. DFR retrains only the last layer and matches SOTA in minutes. Why is your two-stage retraining a better use of compute?
5. Waterbirds worst-group accuracy varies by several points across seeds. How many seeds did you run, and is your improvement inside that band?
6. Your error set has higher precision but the same worst-group accuracy. What does that tell you about the mechanism?
7. What is your random-subset control, and what did it show?
8. Would this transfer to a dataset where you don't have ground-truth group labels? Then how would you know it worked?

**Oral-defense difficulty: 7/10.**

### Project 4 — questions
1. Kumar et al. explained OOD failure via feature distortion. Your CKA metric — is it measuring distortion or just measuring *change*? What would distinguish them?
2. The 2024 unifying PEFT study found methods perform similarly once tuned. Did you tune each method equally hard? Show me the search budgets.
3. At n=1 per class, your error bars must be enormous. How many seeds, and what's your CI?
4. Why Office-Home and not a shift with a known mechanism?
5. Your drift regression has an R². What's the confidence interval on the slope, and does it hold across all five methods separately?

**Difficulty: 6.5/10.**

### Project 3 — questions
1. PatchCore hits 99.6% on MVTec. Why are you working on this dataset at all?
2. Your k=1 results — how many reference-image draws, and what's the spread?
3. AUPRO vs. pixel AUROC — why did you pick one, and what does the other hide?
4. When you swapped in DINOv2, did you retune the coreset ratio and layer selection? If not, is the comparison fair?
5. Your fusion gate uses nearest-neighbour distance. Isn't that the same quantity as the memory-bank score itself? Explain why that isn't circular.

**Difficulty: 7/10.** Question 5 is the killer and many students would not survive it.

### Project 1 — questions
1. What do CLIP's patch tokens actually encode? They were never explicitly supervised at the patch level.
2. APRIL-GAN added linear layers to map image features into the joint space and won the VAND zero-shot track. What does your adapter do that theirs doesn't?
3. Your score map is upsampled from a 14×14 grid to 224×224 and smoothed. How much of your AUPRO gain is the smoothing kernel?
4. You trained the adapter on held-out categories. How do you know it isn't just memorizing "defects look like high-frequency texture"?
5. Why does a *text* encoder help detect a scratch it has never been shown?

**Difficulty: 8/10.** Questions 3 and 5 are genuinely hard.

### Project 7 — questions
1. TTAB showed hyperparameter selection is nearly impossible online. How did you select yours, and doesn't that invalidate the comparison?
2. Your gate has a threshold. Where did it come from, and was any test-corruption information used?
3. TENT collapsed in your runs. Is that a bug, a hyperparameter, or the phenomenon? How do you know?
4. Under what stream conditions does your gate *fail*?
5. What's the source-only baseline, and does your gated method ever fall below it?

**Difficulty: 8.5/10.**

### Project 5 — questions
1. Show me random sampling on every plot. Does your method beat it at every budget?
2. The TMLR paper found uncertainty is competitive from iteration 2 with DINOv2. What's left for your gate to do?
3. AL results are famously irreproducible. Seeds? CIs?
4. This is an ML project. What is the computer vision content?

**Difficulty: 6/10** — but question 4 is fatal in a CV-professor's course.

### Project 9 — questions
1. Your test set is 15 images. What is the confidence interval on your PSNR?
2. PSNR and SSIM correlate poorly with human perception on LLIE. What else did you measure?
3. Your 45K-parameter model beats a 2M-parameter model. Is that a real efficiency result or the small test set?

**Difficulty: 7/10** with question 1 being close to unanswerable.

### Projects 8, 10 — question 1
"This is the title of a CVPR 2024 / ICCV 2025 paper. Walk me through what you added." **Difficulty: 10/10, unrecoverable.**

---

## PART 14 — RESEARCH REPORT POTENTIAL

Rated on the 14-section structure you specified.

| # | Motivation | Related work | Hypothesis | Experimental depth | Discussion potential | Can explain negatives? | Figures/tables | **Narrative** |
|---|---|---|---|---|---|---|---|---|
| 1 | 7 | 8 | 7 | 7 | 6 | 6 | 8 (anomaly maps) | **7.0** |
| **2** | **9** (a live contradiction) | **9** (a clean thesis→antithesis→synthesis arc) | **9** | **9** | **9** | **10** | **9** (severity curves, reliability diagrams, Fourier heatmaps, ANOVA table) | **9.0** |
| 3 | 7 | 8 | 7 | 8 | 7 | 7 | 9 | **7.5** |
| 4 | 8 | 8 | 8 | 8 | 8 | 8 | 6 (mostly line plots) | **8.0** |
| 5 | 6 | 7 | 6 | 7 | 6 | 7 | 5 (budget curves only) | **6.0** |
| 6 | 9 | 9 | 8 | 8 | 8 | **9** | 8 (background swaps, precision/recall plots) | **8.5** |
| 7 | 9 | 9 | 8 | 7 | 8 | 7 | 6 | **8.0** |
| 8 | 8 | 9 | 5 | 3 | 4 | 3 | 9 | **4.5** |
| 9 | 6 | 7 | 5 | 6 | 5 | 5 | **10** | **6.0** |
| 10 | 8 | 9 | 5 | 3 | 4 | 3 | 9 | **4.5** |

**Why Project 2 scores highest on "related work."** Most undergraduate related-work sections are a list. Yours would be an *argument*: paper A claimed X, paper B showed A's evidence was confounded, paper C then showed B's residual explanation was also wrong, and here is the confound none of them removed. That is a related-work section a professor will read twice.

**Why Project 2 scores 10 on "can explain negative results."** Because there is no negative result available. Every outcome of a variance decomposition is a finding. This is structurally different from projects whose report depends on "our method won."

---

## PART 15 — PRESENTATION AND DEMO QUALITY

| # | Visual appeal | Easy to demo? | Interactive potential | Before/after | Masks/maps | Attention viz | Calibration viz | Failure viz | Real-world relevance | 8–12 min explainable? | Memorable? | **Presentation /10** | **Demo /10** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | High | Yes | Medium | Yes | **Yes** | Yes | No | Yes | High | Yes | Yes | **8** | **8** |
| **2** | **High** | **Yes** | **Medium** | **Yes (clean vs. severity-5 side by side)** | No | **Yes (Grad-CAM vs. attention rollout)** | **Yes (reliability diagrams)** | **Yes** | **High** | **Yes** | **Yes (the "both models are 70% accurate but one is confidently wrong" slide)** | **8** | **7** |
| 3 | High | Yes | Medium | Yes | **Yes** | Partly | No | Yes | **Very high** | Yes | Yes | **9** | **9** |
| 4 | Low–medium | Yes | Low | Weak | No | Optional | Optional | Yes | High | Yes | Moderate | **6** | **5** |
| 5 | Low | Yes | Medium (a live "which image next?" viewer) | No | No | No | Yes | Weak | Medium | Yes | Low | **6** | **6** |
| 6 | **High** | **Yes** | **High** (upload a bird photo, swap the background live, watch the prediction flip) | **Yes** | No | **Yes** | Optional | **Excellent** | High | Yes | **Very** | **8.5** | **9** |
| 7 | Medium | Medium | Medium (live error curve as the stream runs) | Yes | No | No | Yes | Yes | High | Hard — needs setup | Moderate | **6.5** | **6.5** |
| 8 | Very high | Weights available | High | Yes | **Yes** | No | Yes (mask diversity) | Yes | High | Yes | Yes | **8.5** | **9** |
| 9 | **Highest** | **Yes** | **High** (drag a slider on a dark photo) | **Yes, dramatic** | No | No | No | Yes | High | Yes | **Very** | **9.5** | **9.5** |
| 10 | Very high | Weights available | **Highest** (click-to-segment) | Yes | **Yes** | No | No | Yes | High | Yes | Yes | **9.5** | **9.5** |

**The uncomfortable truth in this table:** the three projects with the best demos are the three you cannot legitimately do. Do not let demo appeal drive this decision. A stunning demo attached to a reproduced paper is worse than a modest demo attached to real work — the demo makes the reproduction more obvious, not less.

**Project 2's best demo idea** (and it is a good one): a single slide with two images of the same corrupted input, one classified by the CNN and one by the ViT, both wrong, with confidence bars showing the CNN at 94% confidence and the ViT at 41%. Then the reliability diagrams underneath. That slide *is* your H2, and it lands in three seconds.

**Project 6's best demo idea:** a live widget where the audience picks a bird image, swaps land↔water background, and watches ERM flip its prediction while your method doesn't. This is the most memorable demo among the legitimate options.

---

## PART 16 — LONG-TERM VALUE

| # | CV research internships | ML internships | RA applications | Grad applications | Engineering portfolio | Future independent research | Workshop publication realistic? |
|---|---|---|---|---|---|---|---|
| 1 | High | Medium | High | High | Medium | Medium | Possible (VAND-style workshop) |
| **2** | **High** | **High** | **Very high** — "can design a controlled experiment" is the #1 thing RA supervisors screen for | **Very high** | Medium | **High** — the harness is reusable for many follow-ups | **Yes** — an honest confound-control study is exactly a workshop-paper shape (e.g. an ICLR/NeurIPS workshop on robustness or on reproducibility) |
| 3 | High | Medium | High | High | **High** (anomalib-adjacent skills are directly employable in manufacturing/QA) | Medium | Possible |
| 4 | Medium | **High** — PEFT is the single most commercially relevant skill here | High | High | **High** | High | Possible |
| 5 | Low–medium | Medium | Medium | Medium | Medium | Medium | Unlikely (crowded) |
| 6 | **High** | High | **High** | **High** — fairness/robustness is a hot area for admissions | Medium | **High** | **Yes** |
| 7 | High | Medium | High | High | Medium | High | Yes if the harm-map is clean |
| 8 | Medium (as reproduction) | Low | Low | Low | Medium | Low | No |
| 9 | Medium | Low | Medium | Medium | **High** (deployable) | Medium | Unlikely |
| 10 | Medium | Low | Low | Low | Medium | Low | No |

**The RA-application point deserves emphasis.** When a professor takes on an undergraduate RA, the question they are answering is "can this person run an experiment I can trust?" A project whose entire content is *"we found a confound in the literature and controlled for it, and here is the variance decomposition"* answers that question better than a project that produced a slightly better number.

---

## PART 17 — RISK ANALYSIS AND MITIGATION

### Project 2 (recommended)
| Risk type | The risk | Mitigation |
|---|---|---|
| **Technical** | ViT-Tiny trained from scratch on CIFAR-100 underperforms badly, making accuracy-matching impossible | Use **Swin-Tiny** or a hybrid (hierarchical, conv-stem ViT) as the transformer arm — these train far better on small data. Alternatively match at the achievable accuracy and report the residual gap explicitly. Also run Tier A (pretrained models) which sidesteps this entirely. |
| **Research** | Result is "the recipe explains everything," which feels anticlimactic | It isn't — that is the Bai et al. finding *confirmed under a stricter control*, plus your calibration dissociation as the second result. **Pre-register both hypotheses in Week 2** so the report reads as a test, not a fishing trip. |
| **Dataset** | Downloading CIFAR-100-C / Tiny-ImageNet-C fails or is slow | These are small, mirrored, and unrestricted. Download in Week 1 and checksum. Lowest dataset risk on the whole list. |
| **Compute** | No A100 access | Documented fallback: CIFAR-100 @32×32, ViT-Tiny, 3 seeds, 12 runs → fits a single consumer GPU or free Colab in ~20h. |
| **Scope** | Factorial explodes (you add ConvNeXt, then MLP-Mixer, then ImageNet) | **Hard rule: 2 architectures, 3 recipes, 3–5 seeds. Freeze this in Week 2 and do not renegotiate.** Extra architectures go in Tier A (evaluation-only) where they cost nothing. |
| **Evaluation** | Reporting differences that are inside seed noise | Compute the seed std **before** looking at the architecture comparison. Adopt the rule: no claim without a gap exceeding 2× pooled std. |
| **"Sounds impressive but fails"** | It becomes a benchmark table with a fancy title | The four defenses: (1) the factorial and ANOVA, (2) seeds and CIs, (3) the ECE axis, (4) the Fourier mechanism section. Each is a concrete artifact you can point at. |

### Project 6
Technical: worst-group accuracy variance is large — mitigate with 5 seeds and CIs, non-negotiable. Research: persistence may not convert to accuracy — mitigate by making the precision/recall measurement the *primary* deliverable so the accuracy result is a bonus. Dataset: Waterbirds construction has known label-imbalance quirks — mitigate by using the standard distributed splits and citing them. Compute: CelebA at 200k is the only heavy piece — subsample to 50k and state it. Scope: do not add a third dataset. Evaluation: **always show the random-subset error set control.**

### Project 4
Technical: PEFT implementations differ subtly across libraries — mitigate by using one library (`peft` + `timm`) and reporting exact configs. Research: the 2024 unifying study may have pre-empted your ranking result — mitigate by leading with the drift diagnostic, not the ranking. Scope: 5 methods × 4 budgets × 3 seeds × 4 eval sets = 240 evaluations — freeze it. Evaluation: n=1-per-class results are extremely noisy — use 5 seeds at the smallest budgets.

### Project 3
Technical: AUPRO is easy to implement wrong — mitigate by using `anomalib`'s implementation and validating against PatchCore's published MVTec numbers (93.5 PRO) as a correctness check. **This is a great built-in sanity test and you should say so in the report.** Research: MVTec saturation — mitigate by restricting to k≤8. Dataset: MVTec registration delay — start Week 1. Evaluation: k-shot draw variance — 5 draws per category minimum.

### Project 1
Technical: adapter training on frozen CLIP is finicky and the design space is large. Research: heavy crowding; the "what did you add over APRIL-GAN" question. Evaluation: the smoothing confound. Mitigation for all three is the same — run the null controls *first*, in Week 3, before investing in the method.

### Project 7
Technical: **silent divergence.** Mitigate by logging per-batch error and BN-statistic drift for every stream and plotting them; make collapse visible. Research: hyperparameters cannot be selected honestly online (TTAB's own finding) — mitigate by fixing hyperparameters across all corruptions and reporting that you did. Compute: thousands of streams — cut the grid as specified in Part 8. Scope: this project's grid explodes faster than any other on the list.

### Projects 8, 9, 10
The dominant risk is that they cannot be completed (8, 10) or that completion yields no insight (9). No mitigation exists for 8 and 10 within 8 weeks.

---

## PART 18 — 8-WEEK EXECUTION REALITY CHECK

I simulate assuming realistic friction: dependency hell in week 1, one dataset problem, one training run that has to be redone, one week where somebody has midterms, and report writing that takes twice as long as expected.

### Time-to-milestone estimates (all projects)

| # | Working baseline | First useful result | Proposed method running | Ablations | Robustness/generalization | Final analysis | Report | Slides | **Total (weeks)** | **Rating** |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.0 | 3.0 | 4.5 | 5.5 | 6.5 | 7.0 | 1.5 | 0.5 | **~9.5** | **RED/YELLOW** |
| **2** | **1.0** | **1.5** | **3.0** | **4.5** | **5.5** | **6.0** | **1.5 (parallel)** | **0.5** | **~7.0** | **GREEN** |
| 3 | 1.5 | 2.0 | 3.5 | 5.0 | 6.0 | 6.5 | 1.5 | 0.5 | **~7.5** | **GREEN/YELLOW** |
| 4 | 1.0 | 2.0 | 3.0 | 4.5 | 5.5 | 6.0 | 1.5 | 0.5 | **~7.0** | **GREEN** |
| 5 | 1.0 | 1.5 | 3.0 | 4.0 | 5.0 | 5.5 | 1.5 | 0.5 | **~6.5** | **GREEN** |
| 6 | 1.5 | 2.5 | 3.5 | 5.0 | 6.0 | 6.5 | 1.5 | 0.5 | **~7.5** | **GREEN/YELLOW** |
| 7 | 2.0 | 3.5 | 5.0 | 6.0 | 7.0 | 7.5 | 1.5 | 0.5 | **~9.0** | **RED** |
| 8 | 3.0 | 5.0 | — | — | — | — | — | — | **>>8** | **RED** |
| 9 | 2.0 | 3.5 | 4.5 | 6.0 | 6.5 | 7.0 | 1.5 | 0.5 | **~8.5** | **YELLOW/RED** |
| 10 | 3.0 | 5.0 | — | — | — | — | — | — | **>>8** | **RED** |

Note the "Total" column exceeds 8 for projects 1, 7, 9, 8, 10 — because report writing and slides overlap only partially with experiments. **Any project whose experiments end after week 6.5 will produce a rushed report, and the report carries more grade weight than the experiments in almost every course.**

---

### Week-by-week simulation: PROJECT 2 (the recommended one)

**WEEK 1 — Setup and Tier A**
- *S1:* Environment, `timm`, repo scaffold, config system with `architecture` and `recipe` as independent flags. Download CIFAR-100 + CIFAR-100-C, checksum, write the corruption dataloader.
- *S2:* Literature: read Hendrycks & Dietterich, Bai et al., Wang et al. Write the related-work argument arc. Implement mCE + ECE + reliability diagrams and unit-test them on a dummy model.
- *Shared:* Agree the exact factorial. Write it down. Sign it.
- *Deliverable:* repo runs `train.py --arch resnet18 --recipe minimal --seed 0` end to end on CIFAR-100.
- *Dependency:* none. *Risk:* **LOW.**

**WEEK 2 — First results, hypotheses locked**
- *S1:* Train the 2 baseline cells (ResNet-18 minimal, ResNet-18 DeiT-recipe), 1 seed each. Verify accuracy is sane.
- *S2:* Tier A: evaluate 8 pretrained `timm` models on CIFAR-100-C or an ImageNet-C subset. **This gives you a complete, presentable result by end of Week 2.**
- *Shared:* Write the pre-registration: H1, H2, the metrics, the decision rule for "significant."
- *Deliverable:* Tier A table + pre-registration doc committed to the repo (this doc is itself a maturity signal — mention it in the report).
- *Risk:* **LOW.**

**WEEK 3 — The factorial launches**
- *S1:* Launch all CNN cells: {minimal, DeiT, DeiT+AugMix} × 3 seeds = 9 runs.
- *S2:* Get the transformer arm training well. **This is the week's real risk.** Try ViT-Tiny; if clean accuracy is unacceptable, switch to Swin-Tiny. Decide by Friday.
- *Shared:* Implement the accuracy-matching procedure and document it.
- *Deliverable:* CNN arm complete; transformer arm architecture decided.
- *Go/no-go:* **Is the transformer arm within 5 points of the CNN on clean accuracy?** If no → switch to Swin-Tiny or accept and report the gap.
- *Risk:* **MEDIUM.** This is the only medium-risk week in the plan.

**WEEK 4 — Factorial completes, first analysis**
- *S1:* Transformer cells: 9 runs. Buffer for reruns.
- *S2:* Full corruption evaluation on all completed cells; build the results dataframe; ANOVA + variance decomposition; seed-std computation.
- *Shared:* Look at the primary result together. **Checkpoint: do you have an answer to H1?**
- *Deliverable:* mCE table with mean ± std for all 18 cells. **This is the earliest "safe" point — see below.**
- *Risk:* **MEDIUM.**

**WEEK 5 — Calibration and mechanism**
- *S1:* ECE at every severity, before/after temperature scaling (fit on clean val only). Reliability diagrams. Test H2.
- *S2:* Fourier sensitivity heatmaps per architecture. Grad-CAM vs. attention rollout figures.
- *Shared:* Interpret. Does calibration dissociate from accuracy?
- *Deliverable:* H2 answered; mechanism figures drafted.
- *Risk:* **LOW.**

**WEEK 6 — Targeted ablations + generalization**
- *S1:* Architecture-ingredient ablations: patchify stem, kernel size, normalization type — these directly probe Wang et al.'s three claims. 4–6 short runs.
- *S2:* Generalization: Tiny-ImageNet-C transfer OR CIFAR-10→STL-10 natural shift. Error-set overlap analysis between architectures.
- *Shared:* **Buffer.** Any failed run from weeks 3–5 gets redone here.
- *Deliverable:* ablation table; generalization table.
- *Go/no-go:* **Freeze all experiments Friday of Week 6.** Anything not done is a limitation, not a delay.
- *Risk:* **LOW** (because it's buffered).

**WEEK 7 — Writing**
- *S1:* Methods, experimental setup, results sections. All final figures.
- *S2:* Abstract, introduction, related work (the argument arc), discussion, limitations, future work.
- *Shared:* Full read-through and cross-check every number in the text against the results dataframe.
- *Deliverable:* complete draft.
- *Risk:* **LOW.**

**WEEK 8 — Polish, slides, defense prep**
- *S1:* Slides + demo. *S2:* Report revision, README, reproducibility instructions, seed/config release.
- *Shared:* Mock defense using the 10 questions from Part 13. Practice twice.
- *Deliverable:* everything submitted.
- *Risk:* **LOW.**

**EARLIEST SAFE POINT: end of Week 4.** At that moment you have a controlled factorial with seeds, a variance decomposition, and a Tier A pretrained-model study. If literally everything after Week 4 failed, you could still write a respectable report answering H1. **No other project on this list reaches a defensible state that early.**

**Classification: GREEN.**

---

### Compressed week-by-week for the other viable projects

**PROJECT 6 — GREEN/YELLOW.** W1: setup + Waterbirds + ERM baseline. W2: JTT reproduction + DFR (S1) / persistence scorer + error-set constructor (S2). W3: multi-seed ERM runs; precision/recall measurement. W4: persistence reweighting experiments; **safe point** — you have ERM/JTT/DFR/GroupDRO + the precision measurement. W5: CelebA transfer. W6: ablations (s, λ, T) + buffer. W7–8: report + slides. *Main dependency:* S2's persistence scorer needs S1's multi-seed checkpoints — mitigate by having S1 produce seeds 0–2 in W2 so S2 is never blocked. *Risk:* variance.

**PROJECT 4 — GREEN.** W1: setup + frozen backbone + linear probe. W2: all 5 PEFT methods running at full budget; **early result**. W3: budget sweep. W4: shift evaluation; **safe point**. W5: CKA drift instrumentation + regression. W6: ablations + buffer. W7–8: writing. *Excellent parallelization:* S1 owns the adaptation harness, S2 owns evaluation + drift metrics, and they meet at a shared results schema.

**PROJECT 3 — GREEN/YELLOW.** W1: MVTec access (start immediately) + `anomalib` + PatchCore reproduction against published numbers. W2: WinCLIP-style zero-shot scorer. W3: feature caching for all 3 backbones. W4: fusion rules + k-shot protocol; **safe point**. W5: factorial complete. W6: VisA transfer + buffer. W7–8: writing. *Dependency risk:* MVTec registration delay in W1 — mitigate by starting VisA (open AWS) in parallel.

**PROJECT 5 — GREEN** on time, but low ceiling. Everything after the W1 embedding cache is cheap; you will finish early and then be tempted to over-scope. **The risk here is boredom-driven scope creep, not time.**

**PROJECT 1 — YELLOW/RED.** The adapter design space eats weeks 3–5, and the null controls (smoothing) should logically come first but psychologically come last. If the smoothing control kills your result in Week 6, you have no time to pivot.

**PROJECT 7 — RED.** Weeks 3–5 are consumed by making TTA converge stably. The safe point does not arrive until Week 6, leaving no buffer. TTAB's documented difficulty of online model selection means you may not be able to tell a bug from the phenomenon, and that ambiguity is a schedule killer.

**PROJECT 9 — YELLOW/RED.** BasicSR training schedules are long, and each architectural variant needs a full retrain. The safe point is around Week 5–6, but "safe" here means "we have PSNR numbers on 15 images," which is not scientifically safe.

**PROJECTS 8, 10 — RED.** No safe point exists.

---

## PART 19 — TEAM-OF-2 OPTIMIZATION

The design principle: **split by pipeline stage, not by experiment**, so that neither student is ever blocked, and both touch modelling *and* analysis.

| # | Student 1 | Student 2 | Shared | **Parallelization /10** | Notes |
|---|---|---|---|---|---|
| 1 | Adapter architectures + training loop | Memory bank, scoring, AUPRO eval, stratified analysis | Fusion design, controls, writing | **7** | S2 idles until S1's first adapter trains; mitigate by having S2 build the zero-shot pipeline first |
| **2** | **Training harness + all CNN cells + architecture-ingredient ablations** | **Evaluation suite (mCE, ECE, Fourier, Grad-CAM/rollout) + Tier A pretrained study + statistics** | **Factorial design, accuracy-matching protocol, interpretation, report** | **9.5** | **S2 has a full independent workstream (Tier A) from Week 2 and never waits for S1. The two meet at a shared results schema. Both do modelling and both do analysis.** |
| 3 | PatchCore/PaDiM + backbone swaps + feature caching | CLIP zero-shot scorer + fusion rules + AUPRO/eval | Factorial design, VisA transfer, writing | **8.5** | Clean split along the two score families |
| 4 | PEFT harness + all adaptation runs | Evaluation across domains + CKA drift + regression analysis | Budget/shift grid design, writing | **9** | Very clean |
| 5 | Embedding caching + acquisition functions | AL loop + calibration tracking + gate + statistics | Protocol design, writing | **8** | Fine, but the total work is small enough that two students is almost over-staffing |
| 6 | ERM/JTT/GroupDRO training + multi-seed checkpoint production | Persistence scorer + error-set constructor + precision/recall + DFR | Set-size matching protocol, interpretation, writing | **8** | S2 depends on S1's checkpoints in W2–3; solvable by front-loading seeds |
| 7 | TTA algorithm implementations | Stream simulator + gate + sign-prediction evaluation | Protocol design, debugging collapse, writing | **7** | Both students end up debugging the same convergence problem, which is the classic 2-person failure mode |
| 9 | Model architecture + training | Evaluation, no-reference metrics, qualitative comparisons | Loss design, writing | **7.5** | S2 waits for trained models |
| 8, 10 | — | — | — | **3** | Both students end up watching the same download and the same inference script |

**Project 2's 9.5 is the highest on the list and is not an accident** — it comes from the two-tier design. Tier A (evaluate pretrained models) requires zero training and is entirely S2's; Tier B (train the factorial) is entirely S1's. They share only a results schema, which they agree on in Week 1. Neither can block the other. And because S2 owns the *analysis* toolkit (statistics, calibration, Fourier), S2's contribution is intellectually equal to S1's, not a support role. That matters both for the grade and for the fairness of the split.

**Penalty applied:** Projects 8 and 10 score 3/10 because there is genuinely not two people's worth of legitimate work; Project 7 is penalized because its dominant activity — diagnosing why adaptation collapsed — does not divide.

---

## PART 20 — A100 VALUE ANALYSIS

| # | **A100 value** | Model fits? | Fine-tuning realistic? | AMP usable? | Speedup impact | Does it expand what you can test? | Possible without constant access? |
|---|---|---|---|---|---|---|---|
| 1 | **MEDIUM** | Yes, easily (frozen CLIP-B/16 + small adapter) | Yes (adapter only) | Yes | Moderate | Slightly — more adapter variants | **Yes**, after feature caching |
| **2** | **MEDIUM–HIGH** | **Yes, trivially** | **Yes — from-scratch training is the point** | **Yes, big win** | **~4–6× vs. a T4** | **Yes — it buys SEEDS and ABLATION CELLS, which is exactly the right thing to buy** | **Yes.** Documented fallback in Part 5 fits a T4. |
| 3 | **LOW** | Yes | No training required | Yes | Small | Marginally | **Yes** — nearly GPU-free after caching |
| 4 | **MEDIUM** | Yes | Yes (PEFT is light) | Yes | Moderate | Yes — more budgets × seeds | **Yes** |
| 5 | **LOW** | Yes | No | N/A after caching | Negligible | No | **Yes** — runs on a laptop |
| 6 | **MEDIUM–HIGH** | Yes | Yes (ResNet-50 @224 is the heaviest legitimate load here) | Yes, big win | ~4× | Yes — more seeds, and seeds are what this project needs most | **Partly** — CelebA needs subsampling without an A100 |
| 7 | **MEDIUM** | Yes | Source model training only | Yes | Moderate | Yes — more streams | **Yes**, with a reduced grid |
| 8 | **INSUFFICIENT** | Barely | No — weeks of training | Yes | Irrelevant | No | **No** |
| 9 | **MEDIUM** | Yes | Yes | Yes | Moderate | Yes | **Yes** |
| 10 | **INSUFFICIENT** | Barely | No | Yes | Irrelevant | No | **No** |

**The discipline statement for your report and your defense:** *"We used the A100 to increase the number of random seeds and ablation cells, not the model size. Our conclusions are about experimental design, and additional seeds strengthen them in a way a larger backbone would not."* Say exactly this. It is true for Project 2, it is scientifically correct, and it demonstrates that you understand what compute is *for*.

---

## PART 21 — "SOUNDS ADVANCED" VS "ACTUALLY GOOD"

| # | Title inflated? | Methodology genuinely sophisticated? | RQ strong? | Novelty real? | Evaluation convincing? | Conclusions meaningful? | Complexity earns its keep? | Buzzword-heavy? | **Verdict** |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Yes** — "Spatial Adapters" oversells a depthwise conv | Moderate | Moderate | Thin | Yes if controls are run | Moderate | Partly | Moderately ("VLM", "few-shot", "adapters") | **Slightly inflated** |
| **2** | **No — if anything, undersold.** "A Comparative Study" hides that this is a controlled causal-attribution experiment. **Retitle it.** | Yes — factorial design + variance decomposition is real methodology | **Strong** | **Real but methodological** | **Yes** | **Yes** | **Yes** | **No** | **Under-sold; substance exceeds the title** |
| 3 | Somewhat — "From Memory Banks to Foundation Models" is grandiose for a 2×2 | Moderate | Good | Moderate | Yes | Yes | Yes | Mildly | **Mildly inflated, substance is fine** |
| 4 | No — the "To Fine-tune or to Prompt" framing is honest | Moderate–high | Strong | Moderate | Yes | Yes | Yes | No | **Honest** |
| 5 | **Yes** — "Calibration-Gated ... in the Vision Foundation Model Era" is three buzzwords carrying one idea | Moderate | Moderate | Thin | Yes | Moderate | Partly | **Yes** | **Inflated** |
| 6 | No — plain and accurate | Moderate | **Strong** | **Real** | Yes | **Yes** | Yes | No | **Honest, substance exceeds title** |
| 7 | Slightly — "When Adaptation Hurts" is good; the subtitle is fine | **High** | **Strong** | Moderate | **Uncertain — depends on getting clean measurements** | Yes | Yes | Mildly | **Honest but operationally fragile** |
| 8 | N/A — it *is* the paper title | Very high (theirs) | Strong (theirs) | **Zero for you** | N/A | N/A | No | No | **Not your work** |
| 9 | N/A — it *is* the paper title | Moderate | Weak-ish (parameter count on a 15-image test set) | **Zero for you** | Weak (test-set size) | Moderate | **No** | No | **Not your work; evaluation weak regardless** |
| 10 | N/A — it *is* the paper title | Very high (theirs) | Strong (theirs) | **Zero for you** | N/A | N/A | No | No | **Not your work** |

**Penalties applied per your instruction.** Project 5 is penalized for buzzword density relative to idea count. Project 7 is penalized for being difficult in a way that does not convert into insight reliably. Projects 8 and 10 are penalized to the floor for being foundation-model-heavy with no question left open. Project 1 is penalized for a title that promises an architectural innovation and delivers a conv layer.

**Project 2 receives an anti-penalty.** Its problem is the opposite: the title "A Comparative Study" makes it sound like coursework. Fix the title and the perception problem disappears.

---

## PART 22 — COURSE-SCOPE ELIMINATION FILTER

Applying your rejection criteria strictly.

| Criterion | Fails |
|---|---|
| Requires excessive compute | **8, 10** |
| Requires multi-GPU infrastructure | **8, 10** |
| Too many unrelated components | 1 (VLM + adapters + memory banks + fusion), 7 (TTA + gating + stream simulation + protocol critique) |
| Cannot finish in 8 weeks | **8, 10**; borderline **7**, **1** |
| Novelty depends primarily on engineering effort | **9**, 1 |
| Requires theory inappropriate for Intro ML | 7 (online optimization dynamics, collapse analysis) |
| No clean baseline | none — all have baselines |
| No measurable hypothesis | **8, 10** (as reproductions) |
| No convincing evaluation protocol | **9** (15-image test set) |
| Relies on unavailable proprietary systems | none |
| No interpretable experimental story | 5 (weakly — the story exists but is thin and pre-empted) |
| Cannot divide between 2 students | **8, 10**, borderline 7 |

**ELIMINATED OUTRIGHT: 8, 10.**
**HEAVILY PENALIZED: 9, 7, 1.**
**PENALIZED: 5.**
**SURVIVE CLEANLY: 2, 4, 6, and 3 with scope discipline.**

---

## PART 23 — A-GRADE SCORECARD

Weights as specified: A-grade probability 30%, research depth 15%, experimental quality potential 15%, 8-week feasibility 15%, novelty potential 10%, professor appeal 5%, presentation/demo 5%, portfolio value 5%.

**These are expected-outcome scores, not potential scores.** A project's research-depth score reflects the depth you would *actually reach in 8 weeks*, not the depth the topic could support with unlimited time.

| # | A-grade prob (30) | Research depth (15) | Exp. quality (15) | 8-wk feasibility (15) | Novelty (10) | Prof. appeal (5) | Presentation (5) | Portfolio (5) | **FINAL /100** |
|---|---|---|---|---|---|---|---|---|---|
| **2** | **9.0** | 7.5 | **9.0** | **9.5** | 6.0 | **9.0** | 8.0 | 8.0 | **84.5** |
| **6** | 8.0 | 8.0 | 8.0 | 8.0 | **7.0** | 8.5 | 8.5 | 8.0 | **79.5** |
| **4** | 8.0 | 7.5 | 8.5 | **9.0** | 5.5 | 7.5 | 6.0 | 8.5 | **78.0** |
| 3 | 7.5 | 7.0 | 8.0 | 8.0 | 5.5 | 7.5 | 9.0 | 8.0 | **74.8** |
| 5 | 7.0 | 6.5 | 7.0 | 8.0 | 4.5 | 6.0 | 6.0 | 7.0 | **67.3** |
| 7 | 6.0 | 7.5 | 7.0 | 6.0 | 5.0 | 7.5 | 6.5 | 7.5 | **64.5** |
| 9 | 5.5 | 5.5 | 6.5 | 6.0 | 3.5 | 5.5 | 9.5 | 6.5 | **57.8** |
| 1 | 5.0 | 7.0 | 6.0 | 4.0 | 5.0 | 7.0 | 8.0 | 7.0 | **56.5** |
| 8 | 3.0 | 8.0 | 4.0 | 2.5 | 2.0 | 6.5 | 8.5 | 6.5 | **43.5** |
| 10 | 2.5 | 8.0 | 3.5 | 2.0 | 2.0 | 6.5 | 9.5 | 6.5 | **41.0** |

**Note how Projects 8 and 10 score 8.0 on research depth and still finish last.** That is the scoring system working correctly: the underlying research is genuinely deep, but the depth belongs to MIT CSAIL, not to you, and the feasibility and A-grade probability scores dominate the weighting.

---

## PART 24 — EXPECTED VALUE ANALYSIS

Model: **E[A-grade] ∝ Research Upside × Execution Probability × Experimental Quality × Time Feasibility × Professor Appeal**, each on 0–1.

| # | Research upside | Execution prob. | Exp. quality | Time feasibility | Prof. appeal | **Product** | Rank by EV |
|---|---|---|---|---|---|---|---|
| **2** | 0.75 | **0.93** | 0.90 | **0.95** | 0.90 | **0.537** | **1** |
| 4 | 0.72 | 0.88 | 0.85 | 0.90 | 0.75 | 0.364 | 3 |
| 6 | 0.82 | 0.80 | 0.80 | 0.80 | 0.85 | **0.357** | 2 (tie-break below) |
| 3 | 0.70 | 0.82 | 0.80 | 0.80 | 0.75 | 0.276 | 4 |
| 5 | 0.55 | 0.85 | 0.70 | 0.85 | 0.60 | 0.167 | 5 |
| 7 | 0.85 | 0.55 | 0.70 | 0.60 | 0.75 | 0.147 | 6 |
| 1 | 0.65 | 0.50 | 0.60 | 0.45 | 0.70 | 0.061 | 7 |
| 9 | 0.45 | 0.65 | 0.65 | 0.60 | 0.55 | 0.063 | 8 (≈7) |
| 8 | 0.90 | 0.25 | 0.40 | 0.25 | 0.65 | 0.0146 | 9 |
| 10 | 0.92 | 0.20 | 0.35 | 0.20 | 0.65 | 0.0084 | 10 |

*(Projects 4 and 6 land within 2% of each other on raw EV; the weighted scorecard in Part 23 separates them on novelty and professor appeal, which is why Project 6 ranks 2nd overall and Project 4 ranks 3rd.)*

### Why a theoretically more impressive project can have a lower expected A-grade

Look at **Project 7** vs. **Project 2**. Project 7 has the higher research upside (0.85 vs. 0.75) — asking "when does this method hurt?" is a more mature question than "is this comparison confounded?" Yet its expected value is **3.6× lower**.

The mechanism is multiplicative, not additive. Project 7's execution probability of 0.55 is not pessimism; it comes from a published ICML finding that hyperparameter and model selection in this setting is *exceedingly difficult due to online batch dependency*, and from documented catastrophic failures of the exact baselines you would implement. When execution probability and time feasibility both sit near 0.55–0.60, they compound to ~0.33 and drag everything else down with them.

Project 10 makes the point most starkly: research upside 0.92 — the highest on the list — and an expected value of 0.008, because two factors are near zero.

**The general principle:** research quality is *additive* in how impressed a reader is, but *multiplicative* in whether a reader ever sees the result. Under a hard 8-week deadline with two undergraduates, the multiplicative terms dominate. This is precisely why your instruction to rank by expected outcome rather than theoretical potential is the correct instruction — and why I am not choosing Project 7 despite finding it the most intellectually interesting idea on the list.

---

## PART 25 — HEAD-TO-HEAD COMPARISONS

**1 vs 2** — **2 wins.** Decisive factor: **feasibility asymmetry.** P1's method needs weeks 3–5 of adapter design in a subfield with five A\*-venue competitors; P2's method *is* the experimental design, so it exists on day one.
**1 vs 3** — **3 wins.** Same subfield, but P3 requires zero training and has a built-in correctness check (reproduce PatchCore's published PRO). P1's risk is concentrated in the one thing that can't be checked.
**1 vs 4** — **4 wins.** Decisive: P4's hypothesis is grounded in a theory paper with a mechanism (feature distortion). P1's hypothesis is "locality should help," which is an intuition, not a theory.
**1 vs 5** — **1 wins narrowly.** P1 has real CV content and better visuals; P5 is pre-empted and has almost no imagery. But both are weak.
**1 vs 6** — **6 wins.** Decisive: P6's primary deliverable is a *measurement* that exists regardless of whether the method works. P1 needs the method to work.
**1 vs 7** — **7 wins on science, 1 wins on nothing.** Call it **7**, but neither should be chosen.
**1 vs 8** — **1 wins.** P8 is a paper title.
**1 vs 9** — **9 wins on demo, 1 wins on legitimacy.** **1 wins** — P9's title is taken.
**1 vs 10** — **1 wins.** P10 is a paper title.

**2 vs 3** — **2 wins.** Decisive: **headroom.** MVTec's full-shot saturation (99.6%) means P3 must live in the few-shot regime where variance is largest; P2's measurement space is wide open and low-variance.
**2 vs 4** — **2 wins.** Both are feasible and both are well-posed. Decisive: **professor fit and visual content.** P4's figures are line plots; P2's are severity curves, reliability diagrams, Fourier heatmaps, and attention maps. For a CV professor, that difference is worth several grade points. Secondary factor: P4's closest prior work (the 2024 unifying PEFT study) sits uncomfortably close to P4's core comparison, whereas P2's frontier papers left the specific confound unremoved.
**2 vs 5** — **2 wins decisively.** P5 has minimal computer vision content and is pre-empted by a TMLR paper with public code.
**2 vs 6** — **2 wins, but this is the closest call on the list.** P6 has higher genuine novelty (7.0 vs 6.0) and a more memorable demo. Decisive factors for P2: (a) **variance** — worst-group accuracy on Waterbirds swings several points across seeds, so P6 needs 5 seeds to say anything, while P2's clean-accuracy and mCE measurements are far more stable; (b) **parallelization** — 9.5 vs 8.0, because P6's student 2 waits on student 1's checkpoints; (c) **earliest safe point** — Week 4 vs Week 4.5; (d) P6 must contend with DFR, which is a cheaper method that may simply beat everything and flatten the story.
**2 vs 7** — **2 wins decisively.** Decisive: execution probability, 0.93 vs 0.55. P7's own literature documents that its central methodological problem (online model selection) is unsolved.
**2 vs 8** — **2 wins.** P8 is a paper title requiring MegaMedical.
**2 vs 9** — **2 wins.** Decisive: P9's test set is 15 images, so its headline metric carries no statistical weight — and the title is taken.
**2 vs 10** — **2 wins.** P10 is a paper title requiring MegaMedical.

**3 vs 4** — **4 wins.** Decisive: P4's question ("which adaptation, and can I predict the cost?") has a mechanism behind it; P3's is a factorial with a likely-obvious answer (DINOv2 wins).
**3 vs 5** — **3 wins.** Real CV content, better visuals, less pre-empted.
**3 vs 6** — **6 wins.** Decisive: **novelty and the measurement-first design.** P3's most likely result ("better backbone helps more than fusion") is what everyone would guess.
**3 vs 7** — **3 wins.** Decisive: execution probability. P3 trains nothing; P7's failure mode is invisible.
**3 vs 8** — **3 wins.** **3 vs 9** — **3 wins** (legitimacy + evaluation rigour). **3 vs 10** — **3 wins.**

**4 vs 5** — **4 wins decisively.** Decisive: P4 has a mechanism (feature distortion) and a diagnostic contribution; P5 has a heuristic and a stronger competitor with public code.
**4 vs 6** — **6 wins narrowly.** Decisive: **novelty (7.0 vs 5.5) and professor appeal.** P4's ranking question was largely answered by the 2024 unifying study; P6's measurement question is genuinely open, and shortcut learning is more visually and conceptually central to computer vision than PEFT is.
**4 vs 7** — **4 wins.** Decisive: feasibility, 0.90 vs 0.60.
**4 vs 8, 4 vs 9, 4 vs 10** — **4 wins all three.** Legitimacy and feasibility.

**5 vs 6** — **6 wins decisively.** More novelty, more CV content, better demo, better story.
**5 vs 7** — **7 wins on science; 5 wins on feasibility.** Net: **7 wins narrowly** because P5's contribution is too thin to carry a report even when it finishes on time.
**5 vs 8** — **5 wins.** **5 vs 9** — **9 wins on demo, 5 wins on legitimacy → 5 wins.** **5 vs 10** — **5 wins.**

**6 vs 7** — **6 wins.** Decisive: P6's measurement deliverable is robust to method failure; P7's is not, and P7's debugging burden is the worst on the list.
**6 vs 8** — **6 wins.** **6 vs 9** — **6 wins** (legitimacy, hypothesis strength). **6 vs 10** — **6 wins.**

**7 vs 8** — **7 wins.** P7 is at least your own work. **7 vs 9** — **7 wins** on research quality; P9's title is taken. **7 vs 10** — **7 wins.**

**8 vs 9** — **9 wins.** Both titles are taken, but P9 is at least *executable* in 8 weeks; P8 is not.
**8 vs 10** — **8 wins by a hair.** Tyche's model weights are released and its evaluation is somewhat cheaper than MultiverSeg's interaction-simulation protocol. Both are unusable.
**9 vs 10** — **9 wins.** Executable vs. not.

---

## PART 26 — FINAL RANKING

| Rank | Project | Score /100 | A-grade /10 | Research depth /10 | 8-wk feasibility /10 | Novelty /10 | Prof. appeal /10 | Compute risk | Scope risk | Main strength | Main weakness |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **P2 — CNN vs ViT robustness** | **84.5** | **9.0** | 7.5 | **9.5** | 6.0 | **9.0** | **Low** | **Low** | A live contradiction in A\* literature + an uneliminated confound you can actually eliminate; no outcome is uninteresting; Week-4 safe point; 9.5/10 parallelization | Novelty is methodological, not architectural — must be framed correctly or it reads as a benchmark |
| **2** | **P6 — Persistent errors / shortcut learning** | **79.5** | 8.0 | 8.0 | 8.0 | **7.0** | 8.5 | Low | Medium | Highest genuine novelty among feasible options; measurement-first design; best demo among legitimate projects | Worst-group accuracy variance; DFR may flatten the story; S2 waits on S1's checkpoints |
| **3** | **P4 — PEFT under scarcity + shift** | **78.0** | 8.0 | 7.5 | **9.0** | 5.5 | 7.5 | **Low** | Low | Strongest theory grounding (Kumar et al.); highest commercial portfolio value; cleanest parallelization after P2 | The 2024 unifying PEFT study sits very close; weakest visuals |
| 4 | P3 — Memory banks → foundation models | 74.8 | 7.5 | 7.0 | 8.0 | 5.5 | 7.5 | **Low** | Medium | Best demo/visuals among feasible options; near-zero compute; built-in correctness check | MVTec saturation; likely-obvious answer; k-shot variance |
| 5 | P5 — Calibration-gated AL | 67.3 | 7.0 | 6.5 | 8.0 | 4.5 | 6.0 | Low | Low | Trivially feasible; finishes early | Pre-empted by TMLR'24 with public code; almost no CV content; buzzword-heavy title |
| 6 | P7 — Reliability-aware TTA | 64.5 | 6.0 | 7.5 | 6.0 | 5.0 | 7.5 | Medium | **High** | Most sophisticated research question on the list | Documented near-impossibility of honest online model selection; worst debugging burden; poor parallelization |
| 7 | P9 — Multi-prior Retinex LLIE | 57.8 | 5.5 | 5.5 | 6.0 | 3.5 | 5.5 | Medium | Medium | Best visuals of any project | **Title is a CVPR 2026 paper**; 15-image test set makes the headline metric statistically meaningless |
| 8 | P1 — Spatial adapters for VLM AD | 56.5 | 5.0 | 7.0 | **4.0** | 5.0 | 7.0 | Medium | **High** | Genuine industrial relevance; strong visuals | Most crowded subfield; APRIL-GAN already did adapters; smoothing confound could invalidate results in Week 6 with no time to pivot |
| 9 | P8 — Stochastic in-context med-seg | 43.5 | 3.0 | 8.0 | 2.5 | **2.0** | 6.5 | **High** | **High** | — | **Title is verbatim CVPR 2024 (Tyche)**; needs MegaMedical |
| 10 | P10 — Scalable interactive segmentation | 41.0 | 2.5 | 8.0 | **2.0** | **2.0** | 6.5 | **High** | **High** | — | **Title is verbatim ICCV 2025 (MultiverSeg)**; needs MegaMedical + interaction simulation |

### TOP 3
1. **Project 2** — 84.5
2. **Project 6** — 79.5
3. **Project 4** — 78.0

### TOP 5
1. Project 2 (84.5) · 2. Project 6 (79.5) · 3. Project 4 (78.0) · 4. Project 3 (74.8) · 5. Project 5 (67.3)

---

## PART 27 — THE WINNER

# WINNER: PROJECT 2

**"Architectural Robustness: A Comparative Study of CNNs and Vision Transformers Under Corruption and Domain Shift"** — retitled and redesigned in Part 28.

## 1. Why it wins

Five reasons, in order of weight.

**It is the only project where every possible outcome is a result.** The deliverable is a variance decomposition, not a victory. If the recipe explains the gap, you have confirmed and strengthened Bai et al. under a stricter control. If architecture survives recipe-matching, you have contradicted them under that same stricter control. If calibration dissociates from accuracy, that is a second finding. If it doesn't, that is a clean null against a natural extension of Minderer et al. **There is no configuration of nature in which you have nothing to report.** Under a hard deadline, that property is worth more than any amount of novelty.

**It sits on a genuine, live contradiction in top-tier literature.** ICCV'21 and NeurIPS'21 said ViTs are more robust. NeurIPS'21 (Bai et al.) said the comparison was unfair and CNNs catch up with the right recipe — but attributed OOD generalization to self-attention. ICLR'23 (Wang et al.) then said no, three convolutional design choices suffice. The field reversed itself twice inside 18 months. Undergraduates rarely find a question this open that is also this cheap to attack.

**The specific confound is un-eliminated and eliminable by you.** Nobody matched on *achieved clean accuracy*; nobody reported seed variance in these comparisons; nobody measured accuracy and calibration under shift in one controlled design. Those three gaps are closeable with 24 training runs on CIFAR-scale data.

**It is the most feasible project on the list by a wide margin.** Lowest debugging risk (supervised classification fails loudly), lowest dataset risk (CIFAR-C is a checksummed download), highest parallelization (9.5/10 via the two-tier design), earliest safe point (end of Week 4), and the only project with a documented fallback that runs on a free Colab T4.

**It maximally rewards understanding over engineering.** The contribution *is* the experimental design. You cannot produce it by copying a repo, and every question in the oral defense is a question about a decision you made. That is exactly the profile that converts into a 90%+ grade with a CV professor who is probing whether you understand ML or just ran code.

---

## Why each of the others loses

**2. Why Project 3 loses.** MVTec AD is saturated — PatchCore reported up to 99.6% image AUROC in 2022. That forces you into the k≤4 few-shot regime, where results swing wildly depending on which reference images you happen to draw, so you need many draws just to establish a signal. And your most likely finding — that a DINOv2 backbone helps more than a clever fusion rule — is what most people in the field would guess before you started. A correct, well-executed confirmation of the expected answer on a saturated benchmark is a solid B+/A− project, not an A+.

**3. Why Project 4 loses.** It is genuinely good and finishes third for real reasons, not by default. Two things beat it. First, a 2024 unifying study of PEFT in visual recognition already ran LoRA, VPT, adapters and ten more methods on VTAB-1K with systematic per-method hyperparameter tuning plus ImageNet domain-shift robustness, and found the methods perform *similarly* once properly implemented. Your differentiator has to be the CKA-drift diagnostic — which is a good idea, but it is a single idea carrying an entire project, and if the correlation is weak you are left with a ranking that is already published. Second, and bluntly: your figures are line plots. For a professor whose expertise is computer vision, a project with no images in it is a project they cannot fully engage with.

**4. Why Project 5 loses.** It is pre-empted. Gupte et al. (TMLR 2024) evaluated exactly this question — foundation-model features (DINOv2, OpenCLIP) applied to initial pool selection, diversity, and the representative-vs-uncertainty trade-off — found that uncertainty sampling becomes competitive with TypiClust by the second AL iteration, and **released their code**. Your calibration-gate is a modest delta on a heuristic (TCM) that already exists. Worse, this is a computer vision course: after the embedding-caching step, your project never touches an image again. Your professor will ask what the CV content is, and there is no good answer.

**5. Why Project 6 loses.** This is the closest call and I want to be precise about it, because Project 6 is genuinely excellent and I would sign off on it without hesitation. It has *higher* real novelty than the winner. Three things cost it the top spot. First, **variance**: worst-group accuracy on Waterbirds moves several points across random seeds, which means your entire result lives or dies on running enough seeds — and if a persistence-filtered error set improves worst-group accuracy by 1.5 points with a 2-point standard deviation, you have nothing. Second, **DFR**: Kirichenko et al. showed that retraining only the last layer matches or beats SOTA in minutes on one GPU, which means the cheapest possible method may simply beat your more elaborate one and flatten your story into "the simple thing won again." Third, **the JTT authors themselves already reported that error-set group composition matters while specific examples do not** — which is a published prior *against* your primary hypothesis. That makes H1 more likely to fail than P2's H1, and while a failure is still informative, it is a weaker report. Project 6 is my recommended alternative if you have a specific reason to prefer it.

**6. Why Project 7 loses.** It has the best research question on the list and the worst odds of producing clean measurements. The ICML 2023 TTAB paper — the very paper that motivates your project — reports that selecting appropriate hyperparameters, especially for model selection, is *exceedingly difficult* because of online batch dependency, that performance varies enormously with pretrained-model quality, and that methods fail outright on several shift classes, with documented error rates above 76% for TENT under label shift. Those are not obstacles you route around; they are the terrain. Two undergraduates in weeks 3–5 will not be able to distinguish "my implementation has a bug" from "this is the documented phenomenon," and that ambiguity is fatal on a fixed schedule. It is also the worst-parallelizing project here, because its dominant activity — diagnosing collapse — is one person's job that two people end up doing.

**7. Why Project 8 loses.** The title is verbatim the title of **Tyche** (Rakic, Wong, Gonzalez Ortiz, Cimini, Guttag, Dalca), CVPR 2024, pages 11159–11173, from MIT CSAIL. Its contributions are a convolution block enabling interaction among predicted candidates and in-context test-time augmentation for stochasticity. It was trained on MegaMedical — 53 open-access medical segmentation datasets, over 22,000 scans. You cannot train it on one A100 in 8 weeks, which means the only thing you can do is download the released weights and run inference. Your professor will search the title, find the paper in seconds, and conclude you did not do a literature review. There is no recovery from that.

**8. Why Project 9 loses.** The title is verbatim **Multinex** (Brateanu, Mu, Ancuti et al.), CVPR 2026, with public code and pretrained weights. Set that aside for a moment and it still loses on evaluation: LOL-v1's test set is **15 images**. A PSNR difference between two methods on 15 images has a confidence interval so wide that the number carries almost no evidence, and this is a criticism the LLIE community itself makes of the benchmark. Your professor's first question would be "what is the confidence interval on your PSNR," and you would not have a good answer. It has the best demo on the list and I understand the temptation — but a beautiful slider on a dark photo is not a research finding.

**9. Why Project 10 loses.** The title is verbatim **MultiverSeg** (Wong, Gonzalez Ortiz, Guttag, Dalca), ICCV 2025 — same lab as Project 8. It requires MegaMedical-scale training plus a full simulated-interaction training loop with clicks, scribbles, and bounding boxes. Its reported result (53% fewer scribble steps, 36% fewer clicks to reach 90% Dice on unseen tasks) is not something you can approach without their training corpus. It scores 2.0/10 on feasibility — the lowest number in this entire analysis.

**10. Why Project 1 loses.** Three compounding problems. First, **crowding**: WinCLIP (CVPR'23), AnomalyCLIP (ICLR'24), AdaCLIP (ECCV'24), VCP-CLIP, and InCTRL (CVPR'24) all occupy this exact space in a 24-month window. Second, **prior art on your specific idea**: APRIL-GAN won the CVPR 2023 VAND zero-shot track by adding extra linear layers to map image features into the joint embedding space plus multiple memory banks — which is, structurally, "adapters on CLIP for AD." You will be asked what your spatial adapter adds over that, and "it has a depthwise convolution" is a thin answer. Third, and most dangerous, **the smoothing confound**: your score map is upsampled from a coarse patch grid and Gaussian-smoothed, and a meaningful fraction of apparent localization gains in this literature come from that smoothing rather than from the model. The honest control has to be run — and if it kills your result in Week 6, you have no time to pivot. That combination gives it the lowest feasibility score (4.0) of any project that isn't a published paper title.

---

## PART 28 — OPTIMIZING THE WINNING PROJECT

### FINAL PROJECT TITLE

> **Architecture or Recipe? A Matched-Accuracy, Variance-Decomposed Study of Corruption Robustness and Calibration in CNNs and Vision Transformers**

Short form for slides: **"Architecture or Recipe?"**

Why this title: it names the causal question (architecture vs. recipe), names the methodological contribution (matched-accuracy, variance-decomposed), and names both outcome axes (robustness *and* calibration). It promises nothing it cannot deliver. It does not contain the words "novel," "framework," "enhanced," or "via."

### ONE-SENTENCE RESEARCH QUESTION

> When convolutional and transformer image classifiers are trained under an identical recipe and matched on achieved in-distribution accuracy, how much of the reported corruption-robustness gap remains attributable to architecture rather than to training recipe or random seed — and does the same decomposition hold for calibration under shift?

### PRIMARY HYPOTHESIS (H1)

Under matched clean accuracy and an identical training recipe, the architecture factor will account for a **minority** of the explained variance in mean corruption error, with the training-recipe factor accounting for the majority. Operationally: in a two-way ANOVA on mCE with factors {architecture, recipe}, the recipe main effect will exceed the architecture main effect, and the residual architecture effect after recipe-matching will be smaller than the gap observed between each family's default configuration.

**Falsifiable:** if architecture dominates, H1 is rejected and you have contradicted Bai et al. under a stricter control — a stronger result than confirming them.

### SECONDARY HYPOTHESIS (H2)

Accuracy robustness and calibration robustness **dissociate**. Recipe-matching that substantially reduces the mCE gap will **not** proportionally reduce the ECE gap at high corruption severity, and the architecture effect on ECE will remain significant after temperature scaling fitted on clean validation data.

**Grounded in:** Minderer et al. (NeurIPS 2021) reported that calibration decay under ImageNet-C shift is slower for ViT and MLP-Mixer than for other families, both before and after temperature scaling, and argued architecture is a major determinant of calibration. H2 tests whether that survives a controlled recipe manipulation, which their observational study could not do.

**Falsifiable:** if recipe-matching collapses the ECE gap too, then calibration is also recipe-driven — a clean negative that meaningfully qualifies a NeurIPS finding.

### TERTIARY HYPOTHESIS (H3, optional — only if ahead of schedule)

The architectural ingredients identified by Wang et al. (patchified stem, larger kernels, fewer normalization/activation layers) reproduce most of the residual architecture effect at CIFAR scale, meaning "self-attention" is not required to explain it.

### RESEARCH GAP

> The CNN-vs-ViT robustness literature contains a documented reversal — early studies found ViTs inherently more robust; Bai et al. (NeurIPS 2021) showed the comparisons were confounded by scale and training framework; Wang et al. (ICLR 2023) then showed pure CNNs with three specific design changes match or exceed Transformers. Yet no study in this line (a) matches models on *achieved* in-distribution accuracy rather than nominal scale, (b) treats architecture and recipe as crossed factors with replicate seeds so that the gap can be variance-decomposed against seed noise, or (c) measures corruption robustness and calibration-under-shift within the same controlled design. As a result, the field cannot state how much of the robustness gap is architectural, and cannot say whether accuracy robustness and reliability robustness have the same causes.

### PROPOSED CONTRIBUTION

1. **A matched-accuracy fair-comparison protocol.** Models are equated on achieved clean test accuracy (target ±0.5%, with the residual gap reported) rather than on parameter count or nominal "small" designation. Full procedure released.
2. **A variance decomposition of mCE** across architecture × recipe × seed, reported as an ANOVA-style table with effect sizes — making explicit, for the first time in this comparison line, how large the architecture effect is *relative to seed noise*.
3. **A joint robustness–calibration measurement** testing whether the two dissociate under controlled recipe manipulation.
4. **Mechanistic evidence** via Fourier sensitivity heatmaps linking each architecture's per-corruption failures to its frequency-domain sensitivity profile, plus a cross-architecture error-set overlap analysis (do CNNs and ViTs fail on the *same* corrupted images?).
5. **A fully reproducible small-scale harness** where architecture and recipe are independent config flags, released with all seeds and configs.

### BASELINE MODELS

| Rung | Model | Role |
|---|---|---|
| Simple | ResNet-18, minimal augmentation (random crop + horizontal flip), SGD | Establishes the pipeline is correct; the "naive" reference point |
| Strong standard | ResNet-18, full DeiT-style recipe (RandAugment, Mixup, CutMix, label smoothing, AdamW, cosine schedule, longer training) | **The critical rung.** This is the Bai et al. manipulation. Omitting it invalidates everything. |
| Strong modern | Swin-Tiny (primary) or ViT-Tiny (fallback), identical DeiT-style recipe, matched clean accuracy | The architecture arm under fair conditions |
| Reference (Tier A) | 8–12 pretrained `timm` models spanning ResNet / ConvNeXt / DeiT / Swin | Scale-generalization check; no training required |
| Proposed | The full factorial + variance decomposition + ECE + Fourier analysis | The contribution is the analysis, not a new architecture — state this explicitly |

**Why Swin-Tiny over ViT-Tiny as primary:** hierarchical, windowed transformers train far more successfully on small datasets from scratch, which makes accuracy-matching achievable. If you use plain ViT-Tiny on CIFAR-100 from scratch you will likely land 8–12 points below the ResNet and the matching becomes impossible. Keep ViT-Tiny as a secondary arm for the "does the hierarchy matter?" question if time permits.

### PROPOSED METHOD

There is no new model. The method is the **experimental protocol**, consisting of:
1. A single training harness where `--arch` and `--recipe` are orthogonal flags sharing one dataloader, one evaluation path, one logging schema.
2. An accuracy-matching procedure: train each cell to convergence, then select the checkpoint per architecture that minimizes |clean_acc − target|, where target is the mean clean accuracy across architectures within that recipe. Report every residual gap.
3. A pre-registered analysis plan (H1, H2, metrics, significance rule) committed to the repo in Week 2, before the factorial completes.
4. A significance rule stated in advance: **no difference is claimed unless it exceeds 2× the pooled seed standard deviation.**

### PRIMARY DATASET

**CIFAR-100 + CIFAR-100-C.** 50k train / 10k test at 32×32; corrupted test sets with 15 standard corruptions × 5 severities (Hendrycks & Dietterich, ICLR 2019). Free, unrestricted, instantly available, checksummed. Carve 5k from train as clean validation for temperature scaling and model selection; **never touch the corrupted data during training or tuning.**

### SECONDARY DATASET

**Tiny-ImageNet + Tiny-ImageNet-C** (200 classes, 64×64, 15 corruptions × 5 severities) as the scale-generalization check. Fallback if time is short: **CIFAR-10 → STL-10** for a natural (non-synthetic) shift, using the shared class subset.

### EVALUATION METRICS

Clean top-1 accuracy (the matching variable) · mCE and relative mCE · per-corruption error · ECE (15-bin) at each severity, pre- and post-temperature-scaling · reliability diagrams · cross-architecture error-set Jaccard · seed standard deviation on every cell · parameters, FLOPs, inference latency (secondary table).

### MAIN EXPERIMENTS

- **E1 Tier A:** 8–12 pretrained models evaluated on corruption benchmarks. Establishes the phenomenon at scale, completed by Week 2.
- **E2 Factorial:** {ResNet-18, Swin-Tiny} × {minimal, DeiT, DeiT+AugMix} × {3–5 seeds}. Full corruption grid on each cell.
- **E3 Matching:** accuracy-matched subset extracted from E2; the primary H1 test.
- **E4 Calibration:** ECE at all severities, before and after temperature scaling; the H2 test.
- **E5 Decomposition:** two-way ANOVA on mCE and on ECE; bootstrap CIs on all reported differences.

### ABLATION STUDIES

| Ablation | What it isolates | Cost |
|---|---|---|
| A1 Augmentation ladder (none → flip/crop → RandAugment → +Mixup/CutMix → +AugMix) | Which recipe component drives robustness | 5 runs |
| A2 Normalization (BatchNorm / GroupNorm / LayerNorm in the CNN) | Wang et al.'s "reduce norm layers" claim | 3 runs |
| A3 Patchified stem vs. standard ResNet stem | Wang et al.'s claim #1 — highest value/effort ratio on the list | 2 runs |
| A4 Kernel size (3×3 / 7×7 / 11×11 depthwise) | Wang et al.'s claim #2 | 3 runs |
| A5 Training length (100 vs. 200 epochs) | Guards against "the transformer was undertrained" | 2 runs |
| A6 Seed count sensitivity | How many seeds are actually needed for a stable mCE estimate | free (reuse E2) |

A3 and A4 are the intellectual core of the ablation set because they let you adjudicate between "self-attention causes robustness" and "patchification + large kernels cause robustness" — the exact disagreement between Bai et al. and Wang et al.

### ROBUSTNESS / GENERALIZATION EXPERIMENTS

- Full 15 corruptions × 5 severities, with severity-response curves per architecture.
- Held-out corruption discipline: never tune on the 4 designated validation corruptions (speckle noise, Gaussian blur, spatter, saturate) and never touch the 15 test corruptions until final evaluation.
- Cross-dataset: CIFAR-100-C → Tiny-ImageNet-C with no retuning.
- Natural shift: CIFAR-10 → STL-10 on shared classes.

### QUALITATIVE ANALYSIS

Grad-CAM (CNN) vs. attention rollout (Swin/ViT) on the same images at severity 0, 3, 5 · Fourier sensitivity heatmaps per architecture and per recipe · reliability diagrams at severity 0/3/5 · a montage of images the CNN gets right and the transformer gets wrong, and vice versa.

### FAILURE ANALYSIS

Categorize failures by: (a) corruption family (noise / blur / weather / digital); (b) whether the model was confidently or unconfidently wrong (this connects failure analysis to H2); (c) class-pair collapse patterns from confusion-matrix deltas; (d) shared vs. architecture-specific failures via error-set overlap. Report the 10 hardest images per architecture with confidences.

### EXPECTED FIGURES

1. Severity-response curves (accuracy vs. severity, one line per architecture×recipe, shaded seed bands) — **the money figure**
2. Grouped bar chart: mCE by architecture × recipe with seed error bars
3. Variance decomposition bar (share of mCE variance: architecture / recipe / interaction / seed)
4. ECE vs. severity curves, pre- and post-temperature-scaling (two panels)
5. Reliability diagrams at severity 0 and 5 for both architectures (2×2 grid)
6. Fourier sensitivity heatmaps, one per architecture per recipe
7. Per-corruption error radar or heatmap (15 corruptions × architectures)
8. Grad-CAM vs. attention rollout comparison on corrupted inputs
9. Error-set overlap Venn / Jaccard matrix
10. Ablation ladder: mCE vs. augmentation strength

### EXPECTED TABLES

T1 Main results: clean accuracy, mCE, relative mCE, ECE@0, ECE@5 for every cell (mean ± std)
T2 Accuracy-matching table: target, achieved, residual gap per pair
T3 ANOVA / variance decomposition with effect sizes and p-values
T4 Ablation results (A1–A5)
T5 Generalization: Tiny-ImageNet-C and STL-10 transfer
T6 Cost table: parameters, FLOPs, latency, training GPU-hours
T7 Tier A pretrained-model reference table

### EXPECTED VISUALIZATIONS FOR THE DEMO

A side-by-side interactive: pick an image, drag a corruption-severity slider, watch both models' predictions and **confidence bars** change in real time. The moment the CNN stays at 90%+ confidence while being wrong and the transformer's confidence drops is your H2, made visceral in three seconds.

---

## PART 29 — MINIMUM / STRONG / STRETCH SCOPE

I am being deliberately strict. The 8-week deadline dominates.

### MINIMUM VIABLE A-GRADE VERSION
*Target: complete by end of Week 5. This alone earns an A− to A.*

- Dataset: CIFAR-100 + CIFAR-100-C only
- Architectures: ResNet-18 + Swin-Tiny (2)
- Recipes: minimal + DeiT-style (2)
- Seeds: 3
- **Total: 12 training runs**
- Metrics: clean accuracy, mCE, relative mCE, per-corruption error, ECE at severities 0/3/5
- Analysis: accuracy-matching table, mean ± std everywhere, two-way ANOVA on mCE
- Ablations: A1 (augmentation ladder) only
- Figures: 1, 2, 3, 4, 7 from the list above
- Tier A: 6 pretrained models on CIFAR-100-C
- Report: full 14-section structure

This version answers H1 completely and H2 partially. It is a real, defensible, controlled study.

### STRONG VERSION
*Target: complete by end of Week 6. This is the realistic target and earns a solid A.*

Everything in MINIMUM, plus:
- Recipes: add DeiT+AugMix (3 total) → **18 training runs**
- Seeds: 5 on the key cells
- Full ECE at all 5 severities, pre- and post-temperature-scaling → H2 fully answered
- Ablations A2 (normalization) + A3 (patchified stem) → engages Wang et al. directly
- Fourier sensitivity heatmaps → the mechanism section
- Generalization: Tiny-ImageNet-C **or** CIFAR-10→STL-10 (pick one, not both)
- Error-set overlap analysis
- Grad-CAM vs. attention rollout figures
- Interactive demo
- Figures 1–9

### STRETCH VERSION
*Only if you are provably ahead by end of Week 5. Everything here is optional and none of it is required for an A.*

- Ablations A4 (kernel size) + A5 (training length) → full engagement with H3
- A third architecture arm (ViT-Tiny, to test whether hierarchy matters)
- Both generalization datasets
- Tier A extended to ImageNet-C with 12 pretrained models
- Adversarial robustness as a third axis (FGSM/PGD at small ε) — **only if everything else is done**, because it opens a large literature you'd then need to cite properly
- A "recipe transfer" experiment: does giving the CNN the transformer's recipe help more than the reverse?

### The hard scope rules
1. **Freeze the factorial in Week 2.** Written down, both students sign it.
2. **No third architecture in Tier B, ever.** Extra architectures go in Tier A where they cost nothing.
3. **No third dataset.**
4. **All experiments stop Friday of Week 6.** Anything unfinished becomes a "Limitations" bullet, which is a legitimate and expected section of a research report.
5. **If you finish MINIMUM early, add seeds before adding cells.** More seeds strengthen every existing claim; more cells dilute your attention.

---

## PART 30 — DETAILED 8-WEEK PLAN FOR THE WINNER

**S1 = Student 1 (Training & Models). S2 = Student 2 (Evaluation, Statistics & Analysis).**
This split gives both students substantial technical ownership and neither can block the other after Week 1.

---

### WEEK 1 — Foundations
**S1 tasks:** Set up environment (PyTorch, timm, wandb or tensorboard). Build `train.py` with `--arch`, `--recipe`, `--seed`, `--dataset` as fully orthogonal flags. Implement ResNet-18 and Swin-Tiny paths. Implement both recipes. Get one training run to completion on CIFAR-100.
**S2 tasks:** Download and checksum CIFAR-100 and CIFAR-100-C. Build the corruption dataloader. Implement and **unit-test** mCE, relative mCE, ECE (15-bin), and reliability diagrams against hand-computed examples. Read Hendrycks & Dietterich and Bai et al.
**Shared:** Agree the results schema (one row per model×corruption×severity). Agree the factorial. Set up the repo structure and a shared results dataframe format.
**Deliverables:** working `train.py`; tested metrics module; datasets in place; repo scaffold.
**Experiments:** one smoke-test training run.
**Expected result:** ResNet-18 minimal recipe reaches ~72–76% clean accuracy on CIFAR-100.
**Checkpoint:** Does `train.py --arch resnet18 --recipe minimal --seed 0` produce a checkpoint and a metrics row?
**Go/no-go:** If no by Friday, drop Swin-Tiny for now and get the CNN path solid — the CNN arm alone is Week 2's work.
**Potential failure:** dependency/CUDA issues (the classic Week-1 tax).
**Fallback:** use a Colab or Kaggle notebook for Week 1 while fixing the local/cluster environment in parallel.

---

### WEEK 2 — First results and pre-registration
**S1 tasks:** Train ResNet-18 under both recipes, seed 0 each. Verify clean accuracy is sane and the recipe manipulation actually changes something. Begin Swin-Tiny training experiments.
**S2 tasks:** **Tier A study** — evaluate 6–8 pretrained `timm` models on CIFAR-100-C (or an ImageNet-C subset). Produce the first complete results table of the project. Build the plotting module.
**Shared:** Write and commit the **pre-registration document**: H1, H2, metrics, the 2×-pooled-std significance rule, and the planned analysis. Draft the related-work argument arc.
**Deliverables:** Tier A results table + figure; pre-registration committed; related-work draft.
**Experiments:** 2 CNN runs + Tier A evaluation sweep.
**Expected result:** Tier A shows the familiar pattern (transformer-family models with lower mCE), giving you the phenomenon to explain.
**Checkpoint:** **You now have a presentable result.** If the project were cancelled today you would have something.
**Go/no-go:** Does the DeiT recipe measurably improve ResNet-18's mCE vs. minimal? If not, your recipe implementation is wrong — fix before proceeding.
**Potential failure:** the recipes are too similar to produce a measurable effect.
**Fallback:** widen the recipe gap (minimal = crop+flip only; strong = RandAugment + Mixup + CutMix + label smoothing + 200 epochs).

---

### WEEK 3 — Factorial launch (highest-risk week)
**S1 tasks:** Launch all CNN cells: 2–3 recipes × 3 seeds = 6–9 runs. Queue them; they run unattended.
**S2 tasks:** Get the transformer arm to acceptable clean accuracy. **This is the week's real work.** Try Swin-Tiny with the DeiT recipe; if clean accuracy is more than ~5 points below ResNet-18, adjust (longer training, adjusted patch/window size for 32×32 inputs, or upsample CIFAR to 64×64). **Decide by Friday.**
**Shared:** Implement and document the accuracy-matching procedure.
**Deliverables:** CNN arm complete with seeds; transformer architecture decision locked and justified in writing.
**Experiments:** 6–9 CNN runs; 3–5 exploratory transformer runs.
**Expected result:** CNN cells complete; transformer within ~3–5 points on clean accuracy.
**Checkpoint:** Is the transformer arm viable?
**GO/NO-GO — the most important decision point in the project:** if Swin-Tiny cannot get within 5 points of ResNet-18, **switch to upsampling CIFAR-100 to 64×64** (transformers do much better with more tokens) rather than abandoning the arm. Record the change and why.
**Potential failure:** transformer trains poorly on 32×32 from scratch. This is a known, expected difficulty, not a surprise.
**Fallback:** (a) upsample to 64×64; (b) move to Tiny-ImageNet at 64×64 as the primary dataset; (c) worst case, accept the accuracy gap, report it prominently as a limitation, and lean harder on Tier A (pretrained, accuracy-matchable) for the H1 test. **Even the worst case leaves you with a complete project.**

---

### WEEK 4 — Factorial completes; primary result
**S1 tasks:** Transformer cells: 6–9 runs. Rerun any failed CNN cells.
**S2 tasks:** Run the full corruption evaluation across every completed checkpoint. Assemble the master results dataframe. Compute seed standard deviations **first**, then the architecture comparison. Run the two-way ANOVA.
**Shared:** Look at the H1 result together. Write the results-section skeleton around the actual numbers.
**Deliverables:** **T1 main results table** with mean ± std for all cells; **T3 variance decomposition**; Figures 1, 2, 3.
**Experiments:** 6–9 transformer runs; ~18 cells × 75 corruption evaluations.
**Expected result:** an answer to H1, with an effect size and a p-value.
**Checkpoint: ★ EARLIEST SAFE POINT ★** — at this moment you have a controlled factorial with seeds, a variance decomposition, and Tier A. A respectable report could be written from here alone.
**Go/no-go:** if any cell is missing, decide now whether to rerun (Week 6 buffer) or report n=2 seeds for that cell with a note.
**Potential failure:** the architecture effect is entirely inside seed noise.
**Fallback:** **that is a result, not a failure.** "The architecture effect is not distinguishable from seed variance at this scale" is a legitimate, publishable-shaped finding and directly addresses a gap in the literature. Write it up as such, and let H2 carry the second half of the paper.

---

### WEEK 5 — Calibration and mechanism
**S1 tasks:** Ablation A1 (augmentation ladder) — 5 short runs. Begin A2/A3 if time allows.
**S2 tasks:** Full ECE analysis at all severities, pre- and post-temperature-scaling (temperature fitted on clean validation only). Reliability diagrams. Implement the Fourier sensitivity heatmap and run it on each architecture.
**Shared:** Interpret H2. Does calibration dissociate from accuracy? Draft the discussion section.
**Deliverables:** Figures 4, 5, 6; H2 answered; ablation A1 table.
**Experiments:** 5 ablation runs + all calibration/Fourier analyses (cheap, evaluation-only).
**Expected result:** ECE-vs-severity curves that either do or do not separate by architecture after recipe-matching.
**Checkpoint:** Are both hypotheses answered?
**Go/no-go:** if yes → proceed to STRONG scope in Week 6. If H2 is ambiguous → add temperature-scaling variants and stop there.
**Potential failure:** ECE is noisy at high severity because accuracy is near chance.
**Fallback:** report ECE at severities 1–3 where accuracy is still meaningful, and say explicitly why severities 4–5 are excluded from the calibration analysis. That is a correct methodological decision, not a retreat.

---

### WEEK 6 — Ablations, generalization, and buffer
**S1 tasks:** Ablations A2 (normalization) and A3 (patchified stem) — 5 runs. These engage Wang et al. directly. Rerun anything broken.
**S2 tasks:** Generalization: Tiny-ImageNet-C **or** CIFAR-10→STL-10. Error-set overlap analysis. Grad-CAM vs. attention rollout figures.
**Shared:** **This entire week is buffer.** Any failed experiment from Weeks 3–5 gets its rerun here.
**Deliverables:** T4 ablations; T5 generalization; Figures 8, 9.
**Checkpoint: ★ HARD FREEZE ★** — **all experiments stop Friday.** No exceptions. Anything unfinished becomes a Limitations bullet.
**Go/no-go:** freeze regardless of state.
**Potential failure:** the ablations contradict your main result.
**Fallback:** that is genuinely the most interesting outcome available. Lead the discussion section with it.

---

### WEEK 7 — Writing
**S1 tasks:** Methodology, Experimental Setup, and Results sections. Generate every final figure at publication quality (consistent fonts, colorblind-safe palette, error bars on everything).
**S2 tasks:** Abstract, Introduction, Research Question, Hypothesis, Related Work (the thesis→antithesis→synthesis arc), Discussion, Limitations, Future Work.
**Shared:** Full read-through. **Cross-check every number in the prose against the results dataframe.** Every claimed difference must pass the 2×-std rule or be explicitly hedged.
**Deliverables:** complete report draft.
**Checkpoint:** does every figure have a caption that states the takeaway, not just the contents?
**Potential failure:** writing takes longer than expected (it always does).
**Fallback:** the Stretch content is the first thing cut; the Limitations section absorbs it.

---

### WEEK 8 — Polish, demo, defense
**S1 tasks:** Build the slide deck (Part 33 gives the exact sequence). Build the interactive demo.
**S2 tasks:** Report revision pass. README with exact reproduction instructions, all configs, all seeds, environment file. Verify a fresh clone reproduces one cell.
**Shared:** **Mock defense, twice**, using the 10 questions from Part 13. Time the talk to 10 minutes with 2 minutes of slack.
**Deliverables:** final report, slides, demo, repository.
**Checkpoint:** can each student answer any of the 10 questions without the other?
**Potential failure:** the demo breaks live.
**Fallback:** pre-record a 30-second screen capture of the demo and embed it in the slides. Always have this.

---

### Explicit buffer accounting
- Week 6 is **entirely** buffer for experiments.
- Week 4 has slack in S1's schedule for reruns.
- The Minimum scope completes by Week 5, leaving Weeks 6–8 for strengthening and writing.
- Total explicit slack: **~1.5 weeks**, which is the right amount for a plan that assumes imperfect execution.

---

## PART 31 — PAPER READING ROADMAP

Eight essential papers, five supporting, four optional. Not thirty. Read Tier 1 in Weeks 1–2; skim Tier 2 as needed; Tier 3 only if you have spare evenings.

### TIER 1 — ESSENTIAL (read properly, in this order)

**1. Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations," ICLR 2019.**
*What to learn:* the exact definition of mCE and why it is normalized against AlexNet per corruption; the 15 test corruptions vs. the 4 held-out validation corruptions; and — critically — their explicit instruction that networks be trained on clean data and **not** on the corruption images. This paper is your measurement instrument; you must be able to defend every choice in it.

**2. Bai, Mei, Yuille & Xie, "Are Transformers More Robust Than CNNs?", NeurIPS 2021.**
*What to learn:* **the confound argument itself** — that prior comparisons used different scales and different training frameworks. Study their unified training setup in detail; your protocol is a stricter version of it. Note precisely what they concluded (CNNs match ViTs adversarially with the right recipe; ViTs retain an OOD generalization edge they attribute to self-attention) so you can state exactly which of their claims you are testing.

**3. Wang, Bai, Zhou & Xie, "Can CNNs Be More Robust Than Transformers?", ICLR 2023.**
*What to learn:* the three architectural ingredients — patchifying the input, enlarging kernel size, reducing activation and normalization layers — and the fact that they are implementable in a few lines. These become your ablations A2, A3, A4. Also learn *how they framed a contradiction of prior work politely*; you will need that rhetorical move in your discussion.

**4. Minderer, Djolonga, Romijnders, Hubis, Zhai, Houlsby, Tran & Lucic, "Revisiting the Calibration of Modern Neural Networks," NeurIPS 2021.**
*What to learn:* how ECE is computed and its binning sensitivity; the temperature-scaling protocol; and their central findings — that ViT and MLP-Mixer are among the best-calibrated families, that their calibration decays more slowly under ImageNet-C shift both before and after temperature scaling, and that size and pretraining do not fully explain it. This is the entire justification for H2.

**5. Yin, Gontijo Lopes, Shlens, Cubuk & Gilmer, "A Fourier Perspective on Model Robustness in Computer Vision," NeurIPS 2019.**
*What to learn:* how to construct a Fourier heatmap, and the core insight that Gaussian augmentation and adversarial training improve robustness to high-frequency corruptions while *degrading* it on low-frequency ones like fog and contrast. This gives you a *mechanism* for your per-corruption results instead of a list of numbers, and it is the single highest-leverage paper for your discussion section.

**6. Dosovitskiy et al., "An Image Is Worth 16×16 Words," ICLR 2021.**
*What to learn:* patch embedding, positional encoding, and the data-hunger property. You must be able to explain why ViTs need more data or stronger augmentation, because it is the central threat to your experimental design and you will be asked about it.

**7. Touvron et al., "Training Data-Efficient Image Transformers (DeiT)," ICML 2021.**
*What to learn:* the exact recipe — RandAugment, Mixup, CutMix, label smoothing, stochastic depth, AdamW, cosine schedule. This *is* your "strong recipe" arm. Know each component and what it does.

**8. Geirhos, Rubisch, Michaelis, Bethge, Wichmann & Brendel, "ImageNet-Trained CNNs Are Biased Towards Texture," ICLR 2019.**
*What to learn:* the texture-vs-shape framing, and that increasing shape bias improves robustness. This gives you the vocabulary for interpreting *why* one architecture fails on which corruption family.

### TIER 2 — IMPORTANT SUPPORTING (skim; read the results sections carefully)

**9. Bhojanapalli, Chakrabarti, Glasner, Li, Unterthiner & Veit, "Understanding Robustness of Transformers for Image Classification," ICCV 2021.** — *Learn:* the original pro-ViT evidence you are re-examining, and what evaluation choices they made.

**10. Naseer, Ranasinghe, Khan, Hayat, Shahbaz Khan & Yang, "Intriguing Properties of Vision Transformers," NeurIPS 2021.** — *Learn:* the occlusion and patch-permutation experiments; useful as qualitative-analysis inspiration.

**11. Liu, Mao, Wu, Feichtenhofer, Darrell & Xie, "A ConvNet for the 2020s" (ConvNeXt), CVPR 2022.** — *Learn:* which "transformer" design elements are actually portable to CNNs. Directly relevant to your ablations A3/A4.

**12. Liu et al., "Swin Transformer," ICCV 2021.** — *Learn:* windowed attention and hierarchy — why Swin trains better on small data than plain ViT, which justifies your architecture choice.

**13. Guo, Pleiss, Sun & Weinberger, "On Calibration of Modern Neural Networks," ICML 2017.** — *Learn:* the original ECE definition and temperature scaling. Read before Minderer et al.

### TIER 3 — OPTIONAL (only if ahead of schedule)

**14. Zhou, Yu, Xie, Xiao, Anandkumar, Feng & Alvarez, "Understanding the Robustness in Vision Transformers" (FAN), ICML 2022.** — the mid-level-grouping explanation for ViT robustness; a rival mechanism to your Fourier account.

**15. Hendrycks, Mu, Cubuk, Zoph, Gilmer & Lakshminarayanan, "AugMix," ICLR 2020.** — if you use AugMix as a recipe arm, read it.

**16. Kornblith, Norouzi, Lee & Hinton, "Similarity of Neural Network Representations Revisited" (CKA), ICML 2019.** — only if you add a representation-similarity analysis as a stretch item.

**17. Croce et al., "RobustBench," NeurIPS 2021 Datasets & Benchmarks.** — for evaluation-protocol hygiene if you add an adversarial axis.

**Total: 8 essential + 5 supporting.** That is roughly 12–15 hours of reading, which fits Weeks 1–2 comfortably alongside setup. Resist the urge to read more; a related-work section built on eight deeply-understood papers reads far better than one built on thirty skimmed abstracts.

---

## PART 32 — GITHUB REPOSITORY DESIGN

```
architecture-or-recipe/
├── README.md
├── PREREGISTRATION.md
├── environment.yml
├── LICENSE
├── configs/
│   ├── base.yaml
│   ├── arch/          resnet18.yaml, swin_tiny.yaml, vit_tiny.yaml,
│   │                  resnet18_patchstem.yaml, resnet18_k7.yaml, resnet18_groupnorm.yaml
│   ├── recipe/        minimal.yaml, deit.yaml, deit_augmix.yaml
│   └── experiment/    e2_factorial.yaml, e4_calibration.yaml, a1_augladder.yaml, ...
├── src/
│   ├── data/          cifar.py, corruptions.py, tinyimagenet.py, transforms.py
│   ├── models/        builder.py, resnet_variants.py, transformer_variants.py
│   ├── training/      trainer.py, recipes.py, schedulers.py
│   ├── evaluation/    corruption_eval.py, metrics.py, calibration.py,
│   │                  fourier_heatmap.py, error_overlap.py
│   ├── analysis/      anova.py, bootstrap.py, matching.py
│   └── utils/         seeding.py, logging.py, registry.py
├── scripts/
│   ├── download_data.sh
│   ├── run_factorial.sh
│   ├── run_tier_a.sh
│   ├── run_ablations.sh
│   └── make_all_figures.py
├── experiments/
│   └── <exp_id>/      config.yaml, checkpoints/, logs/, metrics.parquet
├── results/
│   ├── raw/           master_results.parquet
│   ├── tables/        T1_main.csv ... T7_tier_a.csv
│   └── stats/         anova_mce.json, anova_ece.json, bootstrap_cis.json
├── figures/           fig01_severity_curves.pdf ... fig10_aug_ladder.pdf
├── notebooks/
│   ├── 01_sanity_checks.ipynb
│   ├── 02_main_analysis.ipynb
│   └── 03_qualitative.ipynb
├── report/            main.tex, refs.bib, figures/ (symlink), main.pdf
├── demo/              app.py, assets/, README.md
└── tests/             test_metrics.py, test_corruptions.py, test_matching.py
```

### What each directory must contain and why

**`PREREGISTRATION.md`** — committed in Week 2, before results exist. States H1, H2, the metrics, and the 2×-pooled-std significance rule. **This single file is one of the strongest maturity signals you can send.** Reference it in your report: "we committed our hypotheses and significance criterion before the factorial completed (see `PREREGISTRATION.md`, commit `abc123`)." Almost no undergraduate project does this, and a professor who notices it will read the rest of your report differently.

**`configs/`** — the architectural heart. Because `arch/` and `recipe/` are separate directories, the orthogonality of your factorial is *visible in the file system*. A reviewer can see at a glance that you did not couple them. Every experiment is fully specified by a config, so reproduction is `python train.py --config configs/experiment/e2_factorial.yaml --arch resnet18 --recipe deit --seed 0`.

**`src/evaluation/metrics.py`** — mCE, relative mCE, per-corruption error. Must be unit-tested (`tests/test_metrics.py`) because a subtle mCE bug silently invalidates every number in the report.

**`src/analysis/matching.py`** — the accuracy-matching procedure as code, not as prose. This is a contribution artifact; make it clean and documented.

**`src/analysis/anova.py`** — the variance decomposition. Outputs `results/stats/anova_mce.json` with effect sizes and CIs, which `make_all_figures.py` reads to produce Figure 3.

**`experiments/<exp_id>/`** — one directory per run, containing its exact config, checkpoint, logs, and metrics. Never overwrite. The `exp_id` should encode arch/recipe/seed so a human can navigate it.

**`results/raw/master_results.parquet`** — the single source of truth: one row per (run, corruption, severity) with all metrics. **Every table and figure is generated from this file by script.** Nothing is hand-copied. When your professor asks "where did this number come from," you point at a row.

**`scripts/make_all_figures.py`** — regenerates every figure from the master results file in one command. This guarantees your report's figures match your data, and it means a late rerun propagates automatically instead of requiring you to remember which of ten figures used the affected cell.

**`tests/`** — three test files is enough. `test_metrics.py` verifies ECE and mCE against hand-computed toy examples. `test_corruptions.py` verifies the corruption loader returns the right shapes and severity ordering. `test_matching.py` verifies the accuracy-matching selection logic. Having tests at all puts you in a small minority of student projects.

**`demo/`** — self-contained, with its own README, so it runs without the training environment.

### README structure
Title and one-paragraph summary → the two hypotheses → headline results table → figure preview → installation → how to reproduce a single cell → how to reproduce everything → repository map → citation of the datasets and the key papers → license. **Put the main results table in the README.** A professor who opens your GitHub should understand what you found in twenty seconds.

---

## PART 33 — FINAL PRESENTATION DESIGN (10 minutes, 12 slides)

Narrative spine: **PROBLEM → WHY IT MATTERS → EXISTING LIMITATION → HYPOTHESIS → METHOD → EXPERIMENT → RESULT → ANALYSIS → FAILURE CASES → CONTRIBUTION → LIMITATIONS → CONCLUSION.**

---

**SLIDE 1 — Title + the hook (0:00–0:45)**
*Purpose:* frame the talk as adjudicating a live disagreement, not reporting a benchmark.
*Content:* Title. Below it, three citations stacked as a timeline: ICCV'21 "ViTs are more robust" → NeurIPS'21 "that comparison was unfair" → ICLR'23 "actually, CNNs can win."
*Visual:* the three-step timeline with arrows.
*Say:* "In eighteen months, this field reversed itself twice. We asked why, and we found a confound nobody removed."
*Answers:* what is this talk about.
*Likely question:* none yet — this is the hook.

**SLIDE 2 — Why corruption robustness matters (0:45–1:30)**
*Purpose:* ground the abstract question in reality.
*Content:* the same image at severity 0 and severity 5 (fog, motion blur), with both models' predictions and confidences.
*Visual:* 2×2 image grid with prediction labels.
*Say:* "Deployed vision systems see rain, motion blur, and bad sensors. A model that's 95% accurate in the lab and 40% in the rain isn't 95% accurate."
*Answers:* why should anyone care.
*Likely question:* "why synthetic corruptions rather than real ones?" — have the answer ready: reproducibility and severity control, plus you also test a natural shift.

**SLIDE 3 — The existing limitation (1:30–2:30)**
*Purpose:* state the gap precisely. This is the most important slide in the talk.
*Content:* three bullets: (1) prior comparisons differ in scale *and* training framework; (2) even the fair-comparison papers match on parameters, not on achieved accuracy; (3) nobody reports seed variance, so we don't know whether these gaps are real.
*Visual:* a schematic showing two models with different clean accuracy being compared OOD, with a red question mark.
*Say:* "If model A is two points worse in-distribution, you cannot attribute its out-of-distribution gap to architecture. That's the confound."
*Answers:* what is unsolved.
*Likely question:* "hasn't Bai et al. fixed this?" — answer: partly, and here's the residual.

**SLIDE 4 — Hypotheses (2:30–3:15)**
*Purpose:* show this is a test, not an exploration.
*Content:* H1 and H2 stated formally, with the note that they were **pre-registered in Week 2**.
*Visual:* two boxed statements; a small git-commit stamp graphic.
*Say:* "We wrote these down before we had results. The commit is in the repo."
*Answers:* what exactly are you testing.
*Likely question:* "what would falsify H1?" — answer it in the slide itself, in one line.

**SLIDE 5 — Method: the factorial (3:15–4:15)**
*Purpose:* show the experimental design is the contribution.
*Content:* the 2×3×5 grid; the accuracy-matching procedure in three lines; the significance rule.
*Visual:* a factorial grid diagram with cells shaded by completion.
*Say:* "There's no new architecture here. The contribution is that architecture and recipe are independent flags in one codebase, with replicate seeds, so we can decompose the variance."
*Answers:* what did you build.
*Likely question:* "why only two architectures?" — answer: because seeds matter more than cells, and we have Tier A for breadth.

**SLIDE 6 — Result 1: the severity curves (4:15–5:15)**
*Purpose:* the primary H1 result.
*Content:* accuracy vs. severity, one line per architecture×recipe, seed bands shaded.
*Visual:* **the money figure.** Make it big; nothing else on the slide.
*Say:* "Under the default recipes, the gap is [X] points. Under the matched recipe, it's [Y] — and the seed band is [Z]."
*Answers:* H1.
*Likely question:* "is [Y] inside the seed band?" — have the number on the slide so you preempt it.

**SLIDE 7 — Result 2: variance decomposition (5:15–6:00)**
*Purpose:* quantify the attribution.
*Content:* stacked bar showing the share of mCE variance attributable to architecture / recipe / interaction / seed, with the ANOVA effect sizes.
*Visual:* one stacked bar, four segments, labelled percentages.
*Say:* one sentence stating which factor dominates.
*Answers:* how much of the gap is architectural.
*Likely question:* "interpret the interaction term physically."

**SLIDE 8 — Result 3: calibration dissociates (6:00–7:00)**
*Purpose:* H2 — and the most memorable slide.
*Content:* left panel, ECE vs. severity by architecture (post-temperature-scaling); right panel, two reliability diagrams at severity 5.
*Visual:* the two-panel figure, plus one corrupted image with two confidence bars — one model 94% confident and wrong, the other 41% confident and wrong.
*Say:* "Both models are equally accurate here. Only one of them knows it's guessing."
*Answers:* H2.
*Likely question:* "why fit temperature on clean data?" — because that's what you'd have in deployment.

**SLIDE 9 — Analysis: why (7:00–7:50)**
*Purpose:* mechanism, not just measurement. This is the slide that separates you from a benchmark report.
*Content:* Fourier sensitivity heatmaps side by side, annotated with which corruption families they predict.
*Visual:* two heatmaps + a small per-corruption error bar chart underneath.
*Say:* "The frequency profiles predict the per-corruption pattern: [architecture] fails on high-frequency noise, [the other] on low-frequency fog and contrast."
*Answers:* why did the result happen.
*Likely question:* "cause or correlate?" — acknowledge honestly that this is associative evidence.

**SLIDE 10 — Failure cases + ablations (7:50–8:40)**
*Purpose:* show you looked for the ways you could be wrong.
*Content:* left, the error-set overlap (do both architectures fail on the same images?); right, the ablation ladder result for patchified stem / normalization.
*Visual:* Venn or Jaccard bar + a small ablation table.
*Say:* "The patchified stem alone recovers [X] of the residual architecture effect — which is evidence for the Wang et al. account over the self-attention account."
*Answers:* what breaks, and which explanation is better supported.
*Likely question:* "did you try larger kernels too?"

**SLIDE 11 — Contributions and limitations (8:40–9:30)**
*Purpose:* be honest before you're asked.
*Content:* left column, four contributions in four words each; right column, three limitations — CIFAR scale not ImageNet, two architecture families only, synthetic corruptions plus one natural shift.
*Visual:* two columns, no images.
*Say:* "We're two students with thirty GPU-hours. Our claims are scoped to this scale, and we say so."
*Answers:* what did you contribute and what don't you claim.
*Likely question:* "does this hold at ImageNet scale?" — point to Tier A as partial evidence.

**SLIDE 12 — Conclusion + demo (9:30–10:00)**
*Purpose:* land it.
*Content:* one sentence per hypothesis, resolved. Then the live demo (or its pre-recorded backup).
*Visual:* the interactive severity slider.
*Say:* "Recipe explains most of the accuracy gap. It does not explain the confidence gap. Those are different problems and should be studied separately."
*Answers:* everything.
*Likely question:* "what next?" — one line: extend to ImageNet scale and to a third architecture family.

### Timing discipline
Slides 6, 7, and 8 are the results and deserve **2 minutes 45 seconds combined**. Slides 1–5 must be done by 4:15. If you're behind, cut slide 2, not slide 9 — the mechanism slide is what earns the top marks.

---

## PART 34 — GRADING STRATEGY

Acting as the grader, with a rubric specific to this project.

| Criterion | Weight | What excellence looks like *here* |
|---|---|---|
| Research question | 10% | Framed as adjudicating a documented reversal in A\* literature, with the specific confound named |
| Literature review | 10% | An argument arc, not a list; the thesis/antithesis/synthesis structure |
| ML understanding | 12% | Correct handling of variance, temperature scaling, train/val/test hygiene; understands why matching on accuracy differs from matching on parameters |
| CV understanding | 12% | Explains texture bias, patchification, receptive fields, frequency sensitivity; interprets per-corruption results mechanistically |
| Methodology | 12% | Orthogonal factorial; documented matching procedure; pre-registration |
| Implementation | 8% | One harness, config-driven, tested metrics, reproducible |
| Experiments | 12% | Complete factorial with seeds; corruption grid; generalization check |
| Ablations | 8% | Each factor removed independently; ablations chosen to adjudicate between competing published explanations |
| Analysis | 10% | Variance decomposition; CIs; error analysis; mechanism |
| Novelty | 6% | Protocol + decomposition + dissociation, honestly framed as methodological |
| Report | — | Folded into the above |
| Presentation | — | Folded into the above |
| Oral defense | — | Folded into the above |

### Grade bands

**60% — Passing.** ResNet and a ViT trained on CIFAR-100, evaluated on CIFAR-100-C, a table of numbers, single seed, and a conclusion of the form "the ViT was more robust." Related work is a list. No ablations. This is a benchmark report.

**70% — Competent.** Both architectures trained under two recipes. Multiple seeds mentioned but variance not used in the argument. mCE computed correctly. One ablation. Related work names the key papers but does not connect them. Conclusions stated without qualification. **This is where most well-executed student projects land.**

**80% — Good.** The full factorial with 3 seeds and error bars. Accuracy-matching attempted and documented. ECE reported alongside accuracy. Related work has an argument. One generalization experiment. Limitations section is honest. Every claimed difference is checked against seed variance. **A solid A−.**

**90% — Excellent.** Everything above, plus: a formal variance decomposition (ANOVA with effect sizes) rather than eyeballed comparisons; the H2 calibration dissociation tested with proper temperature-scaling protocol; ablations chosen specifically to adjudicate between Bai et al. and Wang et al.; a mechanism section using Fourier profiles; and a report where the discussion explains *why* results occurred, not just what they were. The student can defend every design decision orally.

**95%+ — Outstanding.** Everything above, plus: **pre-registration committed before results existed**; a stated significance rule that is actually applied (including places where it forces the students to *not* claim something they'd like to claim); error-set overlap analysis; a clean, tested, config-driven repository where a reader can regenerate every figure with one command; a report whose limitations section identifies weaknesses the grader had not thought of; and a discussion that correctly scopes the claim to CIFAR-scale while using Tier A as partial evidence for broader validity.

### What separates a 70–80% project from a 90–95%+ project — specifically, for this project

It is **not** more experiments. A 70% project and a 95% project may have run the same number of training runs. Five differences do the work:

1. **Variance is used as an argument, not decoration.** The 75% project writes "ResNet: 42.1 mCE, Swin: 39.8 mCE" and concludes Swin is better. The 93% project writes "42.1 ± 1.4 vs. 39.8 ± 1.2; the difference (2.3) exceeds 2× the pooled std (1.8), so we claim it" — and, crucially, somewhere else in the report writes "the difference did *not* exceed our threshold, so we do not claim it." **The willingness to decline a claim you'd like to make is the single clearest marker of scientific maturity.**

2. **The confound is named and removed, not just mentioned.** Many students will write "prior work is confounded" in the introduction and then run an uncontrolled comparison anyway. The matched-accuracy procedure, documented as code, is what turns the sentence into a method.

3. **The report explains *why*, not just *what*.** A per-corruption table is data. A per-corruption table next to a Fourier heatmap, with a paragraph explaining that the high-frequency-sensitive model fails on Gaussian/shot/impulse noise while the low-frequency-sensitive one fails on fog and contrast — that is analysis. This is where your instruction "prefer projects where we can clearly explain WHY results happened" pays off directly.

4. **Ablations adjudicate between published explanations rather than exploring randomly.** Ablating the patchified stem is worth ten times more than ablating the learning rate, because it speaks to a specific disagreement between two ICLR/NeurIPS papers. Choose ablations by *what question they settle*.

5. **Limitations are specific and self-identified.** "Our study is at CIFAR scale; the architecture effect may behave differently at ImageNet scale, and our Tier A evaluation provides only associative evidence on this point" beats "future work could use bigger datasets" by an enormous margin.

---

## PART 35 — FINAL A-GRADE CHECKLIST

Print this. Tick it in Week 8.

| Area | Requirement | Done? |
|---|---|---|
| **Research** | Research question stated in one sentence and framed as adjudicating a documented literature reversal | ☐ |
| | H1 and H2 stated formally, measurably, falsifiably | ☐ |
| | Pre-registration committed before the factorial completed | ☐ |
| | Significance rule (2× pooled std) stated *and applied*, including at least one place where it forced a non-claim | ☐ |
| **Literature** | 8 Tier-1 papers read properly; related work written as thesis→antithesis→synthesis | ☐ |
| | Every claim about prior work is accurate and cited to the correct venue and year | ☐ |
| | No fabricated citations — every reference verified against the proceedings page | ☐ |
| **Dataset** | CIFAR-100 + CIFAR-100-C downloaded, checksummed, splits documented | ☐ |
| | Clean validation split carved from train; corrupted data never touched during training or tuning | ☐ |
| | The 4 designated validation corruptions kept separate from the 15 test corruptions | ☐ |
| | Secondary dataset (Tiny-ImageNet-C or STL-10) evaluated | ☐ |
| **Baseline** | Simple baseline (ResNet-18, minimal recipe) complete with seeds | ☐ |
| | Strong standard baseline (ResNet-18, DeiT recipe) complete — **the critical rung** | ☐ |
| | Strong modern baseline (Swin/ViT, identical recipe) complete | ☐ |
| | Tier A pretrained-model reference study complete | ☐ |
| **Proposed method** | Accuracy-matching procedure implemented, documented, and its residual gaps reported | ☐ |
| | Factorial harness with orthogonal arch/recipe flags, config-driven | ☐ |
| | Variance decomposition (ANOVA) computed with effect sizes | ☐ |
| **Core experiments** | Full factorial complete: ≥2 arch × ≥2 recipes × ≥3 seeds | ☐ |
| | All 15 corruptions × 5 severities evaluated on every cell | ☐ |
| | Clean accuracy, mCE, relative mCE reported with mean ± std | ☐ |
| **Ablations** | Augmentation ladder (A1) | ☐ |
| | At least one architectural-ingredient ablation (patchified stem or normalization) | ☐ |
| | Each factor removed independently, not jointly | ☐ |
| **Robustness / generalization** | ECE at all severities, pre- and post-temperature-scaling | ☐ |
| | Reliability diagrams at severity 0 and 5 | ☐ |
| | Cross-dataset or natural-shift transfer with no retuning | ☐ |
| **Error analysis** | Per-corruption breakdown by family (noise/blur/weather/digital) | ☐ |
| | Confident-vs-unconfident failure split (links to H2) | ☐ |
| | Cross-architecture error-set overlap computed | ☐ |
| | Hardest-images montage with confidences | ☐ |
| **Novelty** | Contribution framed honestly as protocol + decomposition + dissociation, **not** as a new architecture | ☐ |
| | Fourier mechanism section present | ☐ |
| **Reproducibility** | Every figure and table regenerable from `master_results.parquet` by one script | ☐ |
| | All configs and seeds committed; environment file present | ☐ |
| | Metrics unit-tested; fresh clone verified to reproduce one cell | ☐ |
| **Report** | All 14 sections present | ☐ |
| | Every number in prose cross-checked against the results dataframe | ☐ |
| | Every figure caption states the takeaway, not just the contents | ☐ |
| | Limitations are specific and self-identified | ☐ |
| **Presentation** | 12 slides, timed to 10 minutes with 2 minutes slack | ☐ |
| | Severity curves, variance decomposition, and calibration slides are the visual centre | ☐ |
| | Mechanism slide (Fourier) included — do not cut it | ☐ |
| **Demo** | Interactive severity slider works | ☐ |
| | Pre-recorded backup video embedded in the slides | ☐ |
| **Oral defense** | Both students independently rehearsed all 10 questions from Part 13 | ☐ |
| | Both can explain the accuracy-matching rationale in one sentence | ☐ |
| | Both can state what would have falsified H1 | ☐ |

---

## PART 36 — FINAL EXECUTIVE DECISION

**WINNER:**
Project 2 — retitled **"Architecture or Recipe? A Matched-Accuracy, Variance-Decomposed Study of Corruption Robustness and Calibration in CNNs and Vision Transformers"**

**FINAL SCORE:**
84.5 / 100

**A-GRADE POTENTIAL:**
9.0 / 10

**8-WEEK FEASIBILITY:**
9.5 / 10

**RESEARCH QUALITY:**
7.5 / 10 (expected, not theoretical — this is depth actually reachable in 8 weeks)

**NOVELTY:**
6.0 / 10 (methodological and analytical, not architectural — and honestly framed as such)

**PROFESSOR APPEAL:**
9.0 / 10

**COMPUTE RISK:**
Low

**SCOPE RISK:**
Low

---

**WHY IT WINS:**
It is the only candidate where no possible experimental outcome leaves you without a result. The deliverable is a variance decomposition, not a victory, so nature cannot refuse to cooperate. It attacks a genuine, live contradiction — ICCV'21 and NeurIPS'21 said transformers are inherently more robust; Bai et al. (NeurIPS'21) showed those comparisons were confounded by scale and framework; Wang et al. (ICLR'23) then showed pure CNNs with three design changes match or beat transformers — and it removes a confound none of them removed: matching on *achieved* in-distribution accuracy, with replicate seeds so the gap can be judged against noise, and with calibration measured alongside accuracy. It has the lowest debugging risk on the list (supervised classification fails loudly), the lowest dataset risk (a checksummed download), the highest parallelization score (9.5/10, because Tier A and Tier B are fully independent workstreams), the earliest safe point (end of Week 4), and the only documented fallback that runs on a free Colab T4 if the A100 never materializes. And it rewards understanding over engineering: the contribution *is* the experimental design, so it cannot be produced by copying a repository, and every oral-defense question is a question about a decision the students made themselves.

**MAIN RESEARCH QUESTION:**
When convolutional and transformer image classifiers are trained under an identical recipe and matched on achieved in-distribution accuracy, how much of the reported corruption-robustness gap remains attributable to architecture rather than training recipe or random seed — and does the same decomposition hold for calibration under shift?

**PRIMARY HYPOTHESIS:**
Under matched clean accuracy and identical recipes, the training-recipe main effect on mean corruption error will exceed the architecture main effect, and the residual architecture effect will be smaller than the gap observed between each family's default configuration.

**RESEARCH GAP:**
No study in the CNN-vs-ViT robustness line matches models on achieved in-distribution accuracy rather than nominal scale, treats architecture and recipe as crossed factors with replicate seeds so the gap can be variance-decomposed against seed noise, or measures corruption robustness and calibration-under-shift within the same controlled design.

**MAIN NOVELTY:**
A matched-accuracy fair-comparison protocol plus a variance decomposition of mCE and ECE across architecture × recipe × seed — and the resulting test of whether accuracy robustness and calibration robustness dissociate. Methodological, not architectural, and stated as such.

**MAIN RISK:**
The transformer arm trains poorly from scratch on 32×32 CIFAR-100, making accuracy-matching impossible and confounding architecture with trainability.

**HOW WE MITIGATE IT:**
Use Swin-Tiny (hierarchical, windowed) rather than plain ViT-Tiny as the primary transformer arm; make the go/no-go decision by Friday of Week 3; if the gap exceeds 5 points, upsample CIFAR-100 to 64×64 or move to Tiny-ImageNet, both of which give transformers more tokens. In the worst case, accept the residual gap, report it prominently as a limitation, and lean on Tier A — pretrained models, where accuracy-matching is trivially achievable — to carry the H1 test. **Every branch of this decision tree still yields a complete project.**

**MINIMUM SUCCESSFUL VERSION:**
CIFAR-100 + CIFAR-100-C; ResNet-18 and Swin-Tiny; minimal and DeiT recipes; 3 seeds; 12 training runs total. Clean accuracy, mCE, relative mCE, per-corruption error, ECE at severities 0/3/5. Accuracy-matching table, mean ± std throughout, two-way ANOVA on mCE. One ablation (augmentation ladder). Tier A on 6 pretrained models. Complete 14-section report. **Achievable by end of Week 5.**

**STRONG VERSION:**
Add the third recipe (DeiT+AugMix) for 18 runs; 5 seeds on the key cells; full ECE at all severities pre- and post-temperature-scaling; ablations on normalization and patchified stem; Fourier sensitivity heatmaps; one generalization dataset; error-set overlap analysis; Grad-CAM vs. attention rollout figures; the interactive demo. **Achievable by end of Week 6.**

**WHAT WOULD MAKE IT A 95%+ PROJECT:**
Pre-register H1 and H2 in a committed file in Week 2 and reference the commit in the report. State the significance rule in advance and apply it visibly — including at least one place where it forces you to decline a claim you would have liked to make. Choose ablations that adjudicate between Bai et al.'s self-attention explanation and Wang et al.'s patchification-and-kernel explanation, rather than ablating hyperparameters. Include the Fourier mechanism section so the report explains *why*, not just *what*. Make every figure and table regenerable from a single results file with one command. Write a limitations section that identifies a weakness the grader had not thought of, and correctly scope the claim to CIFAR scale while using Tier A as explicitly *associative* evidence for broader validity.

**WHAT WE MUST NOT DO:**
Do not add a third architecture to the trained factorial — extra breadth belongs in Tier A where it costs nothing. Do not add a third dataset. Do not report a single seed for any cell. Do not claim a difference that sits inside 2× the pooled standard deviation. Do not train on the corruption images or tune on the 15 test corruptions. Do not use the A100 to train a bigger model; use it to buy more seeds. Do not let experiments run past Friday of Week 6. Do not describe the contribution as a new method — it is a protocol and an analysis, and overselling it is the fastest way to lose the professor's trust. And do not, under any circumstances, choose a project whose title returns a published paper as the first search result.

---

> **If I were the professor, I would approve this project over the other nine because it is the only one where two undergraduates can run a genuinely controlled experiment — with replicate seeds, a documented matching procedure, and a pre-registered hypothesis — on a question that three top-tier papers have answered three different ways, and hand me a result that is scientifically informative no matter which way it comes out.**
