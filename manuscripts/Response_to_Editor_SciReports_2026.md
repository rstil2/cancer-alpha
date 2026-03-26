# Point-by-Point Response to Editor — Manuscript ID 676329be

---

## Editor Comment 1

> "It would be important for the authors to validate the classifier on more external datasets, more tailored to the ultimate goal, which is to classify cancer samples of unknown primary."

### Response

We thank the editor for this important suggestion, which has substantially strengthened the manuscript.

We have implemented a dedicated CUP validation protocol (new Section 2.6) that directly simulates the clinical scenario described. In each of 10 repetitions with different random seeds, the balanced dataset (n=1,248) was split into training (80%) and test (20%) sets with stratification. The multi-modal LightGBM classifier was trained on the training set and evaluated on held-out test samples that the model has never encountered during training. This yielded 2,500 total predictions.

**Key results (new Section 3.4, Tables 4–5):**
- **97.6% ± 0.9% balanced accuracy** for cancer-of-origin identification
- **99.7% top-3 accuracy** (true type among three highest-probability predictions)
- **99.6% accuracy** when classifier confidence exceeds 90% (86.3% of all predictions)
- Per-type accuracy: BRCA 99.7%, PRAD 99.7%, COAD 99.0%, LIHC 98.4%, STAD 98.4%, HNSC 96.1%, LUAD 95.8%, LUSC 93.6%

The lowest-performing type (LUSC, 93.6%) is consistent with its established molecular overlap with HNSC (shared squamous cell biology; Travis et al. 2015), representing genuine biological ambiguity rather than classifier failure. We discuss the clinical implications and comparison to Moran et al.'s methylation-based CUP classifier (87.7% across 38 types) in new Section 4.2.

We acknowledge in Section 4.6 that further validation on truly external cohorts (ICGC, CPTAC) and actual CUP biopsy samples would further strengthen these findings.

---

## Editor Comment 2

> "It would be interesting and of medical relevance if the authors could also use their classifier to predict cancer subtypes, which could have medical relevance."

### Response

We appreciate this suggestion, which adds an important dimension of clinical utility.

We have added cancer subtype prediction (new Section 2.7) for three cancer types with well-characterized molecular subtypes: breast invasive carcinoma (5 subtypes), lung adenocarcinoma (3 subtypes), and colon adenocarcinoma (4 subtypes). LightGBM classifiers were trained for each cancer type using stratified 5-fold cross-validation with multi-modal features.

**Key results (new Section 3.5, Table 6):**
- **BRCA:** 80.0% ± 5.7% balanced accuracy across 5 subtypes (n=156)
- **LUAD:** 92.3% ± 4.6% balanced accuracy across 3 subtypes (n=165)
- **COAD:** 81.0% ± 2.6% balanced accuracy across 4 subtypes (n=198)

Importantly, top discriminating features include established subtype markers: FOXA1 and AGR2 for breast cancer (luminal/basal markers), and MMP28 and PLAU for lung adenocarcinoma (extracellular matrix markers). This biological concordance confirms that the framework captures subtype-level biology from general multi-modal features without subtype-specific engineering.

We discuss the clinical relevance—breast cancer subtypes guide endocrine versus chemotherapy decisions; CMS subtypes predict immunotherapy response; lung subtypes determine targeted therapy eligibility—in new Section 4.5.

---

## Editor Comment 3

> "I also recommend that the authors combine the two [manuscripts] into 1 manuscript, given the similar content."

### Response

We have fully merged both manuscripts into a single comprehensive paper. The revised manuscript now presents a unified evaluation across two complementary sample regimes:

1. **Full-cohort setting** (n=1,248 balanced TCGA samples, 4,063 multi-modal features, 8 cancer types): 98.4% balanced accuracy with biological validation via SHAP.

2. **Minimal-data setting** (n=158 imbalanced samples, 110 features, 8 cancer types): 95.0% accuracy with LightGBM+SMOTE, outperforming 16-layer transformers by 11.8 percentage points.

Both settings are now presented in a single Methods section (Sections 2.3–2.4) and a unified Results section (Sections 3.1–3.2). This consolidation:
- Eliminates redundancy between the two manuscripts
- Strengthens the study by demonstrating robustness across sample sizes (158 to 1,248)
- Incorporates the systematic model complexity analysis (12 architectures, inverse R²=0.78) into the unified framework
- Adds new CUP validation and subtype prediction analyses

The revised manuscript includes 7 tables, 6 figure legends, and 35 references, with a total length of approximately 6,200 words.

---

We thank the editor for the constructive feedback, which has substantially improved the manuscript. We believe these revisions address all three editorial concerns and are confident the revised manuscript represents a significant contribution to the field.

R. Craig Stillwell, PhD
Campbellsville University
