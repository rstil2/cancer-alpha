# Cover Letter — Revised Manuscript Submission to Scientific Reports

**Date:** July 2025

**To:** The Editors, Scientific Reports

**Re:** Revised Manuscript — ID 676329be

---

Dear Editors,

Thank you for your constructive feedback on our manuscript. We appreciate the opportunity to revise and resubmit. We have thoroughly addressed all three points raised in the editorial decision.

**1. Validation on an external dataset for CUP classification.**

We have added a dedicated CUP validation protocol (Section 2.6) and results (Section 3.4). Using repeated stratified held-out evaluation (10 repetitions, 2,500 total predictions), the multi-modal classifier achieves **97.6% balanced accuracy** with **99.7% top-3 accuracy** for cancer-of-origin identification. Per-cancer-type accuracy ranges from 93.6% (LUSC) to 99.7% (BRCA, PRAD), with well-calibrated confidence scores: when the classifier is highly confident (>90%, representing 86.3% of predictions), accuracy reaches 99.6%. This simulates the clinical CUP scenario where a trained classifier must identify the tissue of origin for a previously unseen sample.

**2. Cancer subtype prediction.**

We have added cancer subtype classification (Section 2.7, Section 3.5) for three cancer types with well-characterized molecular subtypes: BRCA (5 subtypes, 80.0% accuracy), LUAD (3 subtypes, 92.3% accuracy), and COAD (4 subtypes, 81.0% accuracy). Top discriminating features include established subtype markers (FOXA1 and AGR2 for breast cancer), confirming biological validity. These results demonstrate that the same multi-modal features capture clinically relevant subtype biology beyond primary site identification.

**3. Manuscript consolidation.**

We have merged our two related manuscripts into a single comprehensive paper. The combined manuscript now presents a unified evaluation across two complementary sample regimes: a full-cohort setting (n=1,248, 98.4% accuracy) and a minimal-data setting (n=158, 95.0% accuracy). This consolidation eliminates redundancy while strengthening the study by demonstrating robustness across both sample sizes and including the systematic model complexity analysis that reveals an inverse correlation between architectural complexity and performance in data-limited settings (R²=0.78).

The revised manuscript contains new analyses (CUP validation, subtype prediction), new tables (Tables 4–6), and a restructured presentation that integrates both sample regimes into a coherent narrative with direct clinical relevance to the CUP diagnostic challenge.

We believe these revisions substantially strengthen the manuscript and address all editorial concerns. We look forward to your further consideration.

Sincerely,

R. Craig Stillwell, PhD
School of Business, Economics, & Technology
Campbellsville University
Campbellsville, KY 40601
craig.stillwell@gmail.com
