<div style="page-break-before: always;"></div>

# BASE PAPER

**Title:** Are Transformers More Robust Than CNNs?  
**Authors:** Yutong Bai, Jieru Mei, Alan L. Yuille, Cihang Xie  
**Venue:** NeurIPS 2021  

### Why this paper is our base paper

*   The paper highlights unfair comparisons between CNNs and Transformers in robustness evaluations.
*   It systematically studies the role of training frameworks and architecture in robustness, demonstrating that proper training recipes significantly improve CNN robustness.
*   Our project directly builds from this core question.
*   While the base paper focuses on large-scale ImageNet models, our study scales the problem to a highly controlled CIFAR-100/CIFAR-100-C setting suitable for exhaustive factorial analysis.
*   We explicitly evaluate the `architecture × training recipe` interaction effects across multiple fixed seeds.
*   We expand the scope by adding a rigorous calibration (ECE) analysis to test whether accuracy improvements inherently yield calibration improvements under distribution shift.
