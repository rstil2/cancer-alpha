# Revision Strategy: Combined Manuscript for Scientific Reports

**Date:** March 19, 2026  
**Manuscript ID:** 676329be-dcf5-4e3b-b4e2-296796004f19  
**Editor:** Scientific Reports  
**Author:** R. Craig Stillwell, PhD

---

## Editor's Comments (Verbatim)

> Thank you for submitting your manuscript to our journal. Whereas we see the interest of your method, and appreciate the statistical rigour and careful documentation, we believe that a validation on an external dataset, more tailored to the ultimate goal, which is to classify cancer samples of unknown primary, might be needed. Alternatively you might want to use your classifier to predict cancer subtypes, which could have medical relevance. We also noted that you have submitted a similar paper and we'd like to suggest that you combine the two into 1 manuscript, given the similar content. I hope you will consider this option so I look forward to receiving your new submission.

---

## Three Editor Requests

| # | Request | Priority |
|---|---------|----------|
| 1 | **External validation** on a dataset tailored to classifying Cancer of Unknown Primary (CUP), OR cancer subtype prediction | Required |
| 2 | **Cancer subtype prediction** as an alternative/additional clinical application | Suggested alternative |
| 3 | **Merge the two manuscripts** into a single paper | Required |

---

## The Two Manuscripts to Merge

### Manuscript A (Main — "THIS_IS_IT.docx")
- **Title:** "Multi-Modal Cancer Classification Using Integrated Gene Expression, DNA Methylation, and Somatic Mutation Features from The Cancer Genome Atlas"
- **Dataset:** 1,248 TCGA samples (156 per cancer type), 8 cancer types
- **Features:** 4,063 (2,000 gene expression + 2,000 methylation + 63 mutation)
- **Best result:** 98.4% balanced accuracy (LightGBM, held-out test set n=250)
- **Angle:** Multi-modal integration, high accuracy, biological validation via SHAP

### Manuscript B (Companion — `/science/` directory)
- **Title:** "Clinical-Grade AI from Minimal Real-World Data: Rethinking Sample Size Requirements in Precision Medicine"
- **Dataset:** 158 TCGA samples, 8 cancer types (imbalanced: 11–32 per type)
- **Features:** 110 base features across 6 modalities
- **Best result:** 95.0% ± 5.4% balanced accuracy (LightGBM+SMOTE)
- **Angle:** Sample efficiency, inverse complexity–performance relationship, SMOTE for equity, statistical learning theory (VC dimension)
- **External validation:** 76 ICGC ARGO samples (92.1% accuracy)

### Overlap Between Papers
- Same TCGA source, same 8 cancer types, same LightGBM classifier family
- Both include SHAP interpretability and biological validation
- Both argue against deep learning transformers for this task
- Both emphasise zero synthetic data (except SMOTE in Manuscript B)

---

## Proposed Combined Manuscript Structure

### Proposed Title
**"Multi-Modal Cancer Classification from Abundant and Minimal Data: Achieving Clinical-Grade Accuracy Across Sample Regimes with External Validation"**

*(Alternative: "From 158 to 1,248 Samples: Multi-Modal Cancer Classification Achieving Clinical-Grade Accuracy with External Validation on Cancer of Unknown Primary")*

### Narrative Arc
The combined paper tells a stronger story than either paper alone:

1. **Full-scale performance** (Manuscript A): Given adequate real data (n=1,248), multi-modal integration achieves 98.4% accuracy — no synthetic augmentation needed
2. **Sample efficiency** (Manuscript B): Even with minimal data (n=158), gradient boosting with SMOTE achieves 95.0% — outperforming transformers by 11.8 pp
3. **NEW: External validation** on a CUP-relevant cohort (addresses editor's primary concern)
4. **Practical implication:** Clinical-grade cancer classification is feasible across a range of real-world data availability scenarios

### Proposed Sections

#### Abstract (~250 words)
Combine key results from both sample regimes + external validation result.

#### 1. Introduction
- Cancer classification challenge and CUP problem (from Manuscript A)
- The "big data" assumption and why it matters for clinical translation (from Manuscript B)
- Gap: No study has systematically shown performance across sample regimes with external validation

#### 2. Methods
- **2.1 Data sources and preprocessing**
  - Full TCGA cohort (n=1,248, balanced)
  - Reduced cohort (n=158, imbalanced — realistic clinical scenario)
  - External validation cohort (NEW — see options below)
- **2.2 Feature extraction** (multi-modal: expression, methylation, mutations)
  - Full feature set (4,063 features for large cohort)
  - Reduced feature set (110 features for small cohort)
- **2.3 Classification models** (LightGBM, XGBoost, RF, LR + transformer baselines)
- **2.4 SMOTE for class imbalance** (applied only in small-n regime)
- **2.5 Validation strategy** (stratified 5-fold CV, held-out test, external test)
- **2.6 Interpretability** (SHAP, pathway enrichment)
- **2.7 Statistical analysis** (VC dimension, bias-variance, calibration)

#### 3. Results
- **3.1 Full-cohort classification** (98.4% accuracy, per-class breakdown)
- **3.2 Minimal-data classification** (95.0% accuracy, learning curves)
- **3.3 Model complexity vs. performance** (inverse relationship, R²=0.78)
- **3.4 Effect of SMOTE on clinical equity** (+7.7 pp, minority class rescue)
- **3.5 External validation** (NEW — key addition)
- **3.6 Biological interpretability** (SHAP features, pathway enrichment)

#### 4. Discussion
- Clinical significance: robust across sample regimes
- CUP application and external validation implications
- Simple models outperform complex ones in data-limited settings
- Limitations and future work

#### Figures (target 6–8)
1. Study design overview (data flow for both cohorts + external)
2. Full-cohort performance (confusion matrix, per-class metrics)
3. Minimal-data performance + learning curves
4. Model complexity vs. accuracy (inverse relationship)
5. SMOTE impact on per-class equity
6. External validation results (NEW)
7. SHAP biological interpretability
8. VC dimension / bias-variance analysis (could move to Supplementary)

---

## Addressing the External Validation Request

The editor's primary scientific concern: **validation on a dataset relevant to Cancer of Unknown Primary (CUP).**

### Option 1: ICGC ARGO Dataset (Already Available)
Manuscript B already reports 92.1% accuracy on 76 ICGC ARGO samples. This could be expanded:
- **Pros:** Already done; independent of TCGA; cross-population
- **Cons:** Small (n=76); not specifically CUP samples; editor may want something more tailored

### Option 2: TCGA CUP-Like Simulation
Hold out entire cancer types from training, then predict them as "unknown primaries":
- Train on 7 types, predict the 8th → repeat for each type (leave-one-cancer-out)
- **Pros:** Directly addresses CUP scenario; no new data needed; strong experimental design
- **Cons:** Still TCGA data (not truly external); editor specifically said "external dataset"

### Option 3: GEO/ArrayExpress External Cohort
Download independent gene expression + methylation data from GEO for the same cancer types:
- Potential datasets: GSE62944 (TCGA reprocessed — may not count), various GEO cancer datasets
- **Pros:** Truly external; different platform/batch effects test robustness
- **Cons:** Multi-modal data (expression + methylation + mutations) rarely available together externally; may need to validate on single modality

### Option 4: CPTAC (Clinical Proteomic Tumor Analysis Consortium)
CPTAC has multi-omic data for several cancer types that overlap with TCGA:
- Available: BRCA, LUAD, COAD, LIHC, HNSC
- **Pros:** Truly independent; multi-omic; high quality; publicly available
- **Cons:** Not all 8 cancer types covered; proteomics modality differs

### Option 5: Cancer Subtype Prediction (Editor's Alternative)
Instead of CUP validation, predict molecular subtypes within cancer types:
- BRCA: Luminal A, Luminal B, HER2+, Basal-like, Normal-like
- LUAD: TRU, PP, PI subtypes
- COAD: CMS1–4 consensus molecular subtypes
- **Pros:** High medical relevance (subtypes guide treatment); uses existing TCGA data; editor explicitly suggested this
- **Cons:** Requires subtype labels from TCGA (available for most types); different problem framing

### RECOMMENDED APPROACH: Do Both — CUP Simulation (Option 2) + Subtypes (Option 5)

Having reviewed the existing pipeline code (`src/pipeline/step4_train_evaluate.py`), data (`data/production_tcga/` with all 33 TCGA cancer types available), and the editor's wording, here is the recommendation:

**Do both.** They serve complementary roles and the editor explicitly mentioned both. Together they make the merged manuscript substantially stronger than either alone.

#### Why CUP Simulation (Option 2) is the higher priority:
- The editor's **first** and most detailed comment is about CUP — it's what they care about most
- The existing pipeline already has the data and code infrastructure — this is a relatively small modification to `step4_train_evaluate.py` (train on 7 types, test on 8th, repeat)
- It directly validates the clinical claim both manuscripts make
- Your 33 available TCGA cancer types in `data/production_tcga/clinical/` mean you could even extend beyond the current 8 types for a more impressive CUP test
- **No new data acquisition needed** — can be done immediately

#### Why Subtypes (Option 5) adds significant value:
- The editor explicitly said "you might want to use your classifier to predict cancer subtypes, which could have medical relevance" — doing this shows you took their advice seriously
- Subtype classification is a **harder** problem that tests whether the features capture finer-grained biology
- TCGA subtype labels are freely available via the GDC API or TCGAbiolinks for BRCA (PAM50), COAD (CMS1-4), and LUAD
- Even moderate performance here (80-90%) would be publishable and clinically relevant since subtypes directly guide treatment choices
- **Modest additional effort** — download subtype labels, reuse existing feature matrices, train per-cancer subtype models

#### Why NOT to pursue a truly external dataset (Options 3/4):
- Multi-modal data (expression + methylation + mutations for the same patients) is extremely rare outside TCGA/ICGC
- CPTAC covers only 5 of your 8 cancer types
- GEO datasets are typically single-modality
- The editor said "external dataset" but the CUP simulation + subtypes together demonstrate generalization more convincingly than a small mismatched external cohort
- The existing ICGC ARGO result (n=76, 92.1%) from Manuscript B already provides some external validation

#### Execution order:
1. **CUP simulation first** (can start immediately with existing code/data)
2. **Subtype prediction second** (needs subtype label download, then reuses pipeline)
3. **ICGC ARGO in supplementary** (already done)

#### What the pipeline already supports vs. what needs building:

| Component | Status | Effort |
|-----------|--------|--------|
| Multi-modal feature matrices (4,063 features) | ✅ Exists in `data/real_model_results/` | None |
| LightGBM training + evaluation | ✅ Exists in `step4_train_evaluate.py` | None |
| SHAP analysis | ✅ Exists in `step4_train_evaluate.py` | None |
| Leave-one-cancer-out loop | ❌ Needs building | ~100 lines of new code |
| TCGA subtype label download | ❌ Needs building | ~50 lines (GDC API call) |
| Subtype classifier training | ❌ Needs building | ~150 lines (adapts existing pipeline) |
| Results tables/figures for CUP | ❌ Needs building | ~100 lines (matplotlib) |
| Results tables/figures for subtypes | ❌ Needs building | ~100 lines (matplotlib) |

---

## Merging Strategy: What to Keep, Cut, and Add

### Keep from Manuscript A
- Full multi-modal feature extraction pipeline (4,063 features)
- LightGBM + 3 comparator classifiers on full cohort
- Per-class performance breakdown and confusion matrix
- SHAP analysis with biological validation
- Bayesian hyperparameter optimization details

### Keep from Manuscript B
- Sample efficiency analysis (learning curves)
- Inverse complexity–performance relationship (12 architectures)
- SMOTE analysis for class imbalance / clinical equity
- VC dimension and bias-variance decomposition
- ICGC ARGO external validation (n=76)

### Cut (to avoid redundancy)
- Duplicate methods descriptions (merge into one Methods section)
- Duplicate SHAP analyses (keep one unified version)
- Redundant introduction material
- Duplicate figures showing similar results at different scales

### Add (NEW content)
- CUP simulation experiment (leave-one-cancer-out)
- Cancer subtype prediction (BRCA, COAD, LUAD subtypes)
- Unified discussion connecting both sample regimes to clinical deployment
- Revised abstract and introduction framing the combined contribution

---

## Action Items

### Phase 1: New Experiments
- [ ] **CUP simulation:** Write `experiments/cup_simulation.py` — leave-one-cancer-out cross-validation on the 1,248-sample cohort. Reuse feature matrices from `data/real_model_results/` and model training code from `src/pipeline/step4_train_evaluate.py`. Report top-1 and top-3 accuracy per held-out cancer type, confidence scores, and misclassification patterns.
- [ ] **Subtype labels:** Download TCGA molecular subtype annotations via GDC API or TCGAbiolinks for BRCA (PAM50: LumA, LumB, HER2, Basal, Normal), COAD (CMS1–4), LUAD (TRU, PP, PI). Save as CSV with `patient_id, cancer_type, molecular_subtype`.
- [ ] **Subtype prediction:** Write `experiments/subtype_prediction.py` — train per-cancer LightGBM classifiers using existing multi-modal features. Evaluate with stratified 5-fold CV + SHAP analysis to identify subtype-discriminating features.
- [ ] **Expand CUP to more cancer types:** Current model uses 8 types, but `data/production_tcga/` has 33 TCGA projects. Consider expanding to 15–20 types for a more comprehensive CUP test.

### Phase 2: Manuscript Merging
- [ ] Draft combined Methods section (unify data description, feature extraction, classifiers from both manuscripts)
- [ ] Draft combined Results: full-cohort (98.4%), small-cohort (95.0%), CUP simulation, subtype prediction, ICGC ARGO external
- [ ] Create unified figure set (target 6–8 main figures + supplementary)
- [ ] Write new Introduction framing: "multi-modal classification across sample regimes with CUP and subtype validation"
- [ ] Write new Discussion addressing CUP relevance, subtype clinical utility, and sample efficiency

### Phase 3: Polish and Submit (no deadline — take the time to do it right)
- [ ] Internal consistency check (all numbers match across sections)
- [ ] Reference list consolidation (merge both bibliographies, remove duplicates)
- [ ] Format for Scientific Reports guidelines
- [ ] Write new cover letter addressing all three editor comments explicitly
- [ ] Prepare response-to-editor letter

---

## Code and Data Inventory

### Existing Pipeline (ready to use)
- **Feature extraction:** `src/pipeline/step1–step3` (expression, methylation, mutations)
- **Training/evaluation:** `src/pipeline/step4_train_evaluate.py` (LightGBM + Optuna, XGBoost, RF, LR)
- **Multi-modal feature matrices:** `data/real_model_results/` (pickle files, ~4,063 features)
- **Raw TCGA data:** `data/production_tcga/` (33 cancer types: expression, methylation, mutations, clinical)
- **SHAP analysis:** Built into `step4_train_evaluate.py`

### Needs Building
- `experiments/cup_simulation.py` — leave-one-cancer-out CUP experiment
- `experiments/subtype_prediction.py` — within-cancer subtype classifier
- `experiments/download_subtype_labels.py` — GDC API subtype label retrieval
- Updated figures for CUP and subtype results

---

## Draft Response to Editor (for Cover Letter)

> Dear Editor,
>
> Thank you for your constructive feedback on our manuscript (ID: 676329be). We appreciate your recognition of the statistical rigour and documentation, and we have carefully addressed each of your suggestions.
>
> **1. External validation for Cancer of Unknown Primary (CUP):** We have added two new validation experiments. First, a leave-one-cancer-out cross-validation that simulates the CUP scenario by training on 7 cancer types and predicting the held-out type as "unknown." Second, we demonstrate our classifier's ability to predict clinically relevant cancer subtypes (BRCA PAM50 subtypes, COAD consensus molecular subtypes, and LUAD molecular subtypes). We additionally report cross-cohort validation on [N] independent ICGC ARGO samples.
>
> **2. Cancer subtype prediction:** As suggested, we now include subtype classification results demonstrating the medical relevance of our approach for treatment-guiding molecular stratification.
>
> **3. Manuscript consolidation:** We have combined our two submissions into a single, comprehensive manuscript. The combined paper presents cancer classification performance across two complementary sample regimes (n=1,248 balanced cohort and n=158 minimal-data cohort), demonstrating that clinical-grade accuracy is achievable across a range of data availability scenarios. This consolidation strengthens the contribution by showing both the ceiling performance and the practical lower bound.
>
> We believe the revised manuscript now provides [the external validation / subtype classification / consolidated analysis] that addresses your concerns while presenting a more complete scientific contribution.
>
> Sincerely,
> R. Craig Stillwell, PhD

---

## Decisions (Resolved)

| # | Question | Answer |
|---|----------|--------|
| 1 | Which validation approach? | **Both:** CUP simulation (leave-one-cancer-out) + cancer subtype prediction (BRCA, COAD, LUAD) |
| 2 | Analysis code available? | **Yes.** Pipeline in `src/pipeline/`, data in `data/production_tcga/` and `data/real_model_results/` |
| 3 | Resubmission deadline? | **None.** No rush — do it right |
