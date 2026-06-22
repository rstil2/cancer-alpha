# Anticipated Reviewer Response (Draft)

**Manuscript:** Experimental Design Dominates Model Architecture in Multi-Modal Cancer Classification  
**Status:** DRAFT — do not submit until JBI invites revision  
**Use:** Pre-written responses for likely reviewer themes; customize when real comments arrive.

---

## Cover letter (when submitting revision)

Dear Editor,

Thank you for the opportunity to revise our manuscript. We have addressed all reviewer and editor comments point-by-point. Major changes include expanded limitations on external validation, added ethics and author contribution statements, corrected administrative disclosures, softened clinical-deployment language, clarified the canonical reproduction pipeline, and added a planned imbalance-robustness analysis. We believe the revised manuscript is substantially strengthened.

Sincerely,  
R. Craig Stillwell, PhD

---

## Anticipated Comment 1: Novelty — "This confirms Grinsztajn et al.; what is new?"

**Response:** We agree that tree-based superiority on tabular data is established in general ML. Our contribution is a **translational bioinformatics** framing across two clinically realistic regimes (small imbalanced n=158 with SMOTE vs large balanced n=1,248 without synthesis), unified by VC-informed sample-complexity analysis on **multi-modal TCGA genomics**, with external validation in Study 1 (ICGC ARGO) and explicit comparison to deep tabular architectures (TabTransformer, MLP) under identical preprocessing. We have softened language implying a paradigm shift (Abstract, Discussion, Conclusion) and expanded Related Work to position against Li et al. (2017) and Mostavi et al. (2020) TCGA benchmarks.

**Manuscript changes:** Related Work §2.4; Limitations; tone-down throughout.

---

## Anticipated Comment 2: External validation insufficient

**Response:** We acknowledge that Study 2 performance is on a held-out **TCGA** partition only. We have revised Limitations §5.1 to state this explicitly and removed language suggesting near-clinical deployment. Study 1 provides independent ICGC ARGO validation (n=76, 92.1%) but with a different feature set. We agree that CPTAC or additional ICGC cohorts would strengthen generalizability and list this as priority future work. We did not claim CUP biopsy validation in the submitted manuscript.

**Manuscript changes:** Limitations §5.1; Conclusion; Data Availability.

---

## Anticipated Comment 3: Balanced subsampling inflates accuracy

**Response:** Study 2 intentionally isolates the effect of experimental design (balance, leakage control, multi-modal integration) from architecture choice. We added §3.8 describing evaluation on naturally imbalanced TCGA test partitions (`imbalance_stress_test.py`). [Insert actual results when available.] We also note that Study 1 used naturally imbalanced data with SMOTE, bracketing both design choices.

**Manuscript changes:** New §3.8; Limitations §5.1.

---

## Anticipated Comment 4: Repository / reproducibility confusion

**Response:** The repository contains legacy demo, scaling, and transformer experiments not used for manuscript results. We clarified Data Availability to point to the **canonical pipeline** (`src/pipeline/steps 1–4`) and added `science/jbi_revision/supplementary/REPRODUCTION_GUIDE.md`. Study 2 metrics are archived in `data/real_model_results/model_results.json` (LightGBM test balanced accuracy 98.4%).

**Manuscript changes:** Data and Code Availability; supplementary REPRODUCTION_GUIDE.md.

---

## Anticipated Comment 5: Study 1 and Study 2 not directly comparable

**Response:** We agree. The revised Limitations state that Studies 1 and 2 differ in sample size, features (110 vs 4,063), cancer-type panel (KIRC vs LUSC), and imbalance handling (SMOTE vs real balancing). They are **complementary regimes**, not a single cohort analyzed twice. The unified claim is that experimental design dominates architecture in **both** regimes tested.

**Manuscript changes:** Limitations §5.1; unchanged Methods structure with explicit KIRC/LUSC note already in §3.7.

---

## Anticipated Comment 6: VC dimension / theory is hand-wavy

**Response:** VC bounds are presented as interpretive support aligning with empirical bias-variance decomposition, not formal generalization certificates. We added explicit language in Limitations that bounds are approximate. We are willing to move extended theory to Supplementary Methods if reviewers prefer a more empirical focus.

**Manuscript changes:** Limitations §5.1.

---

## Anticipated Comment 7: SMOTE vs "no synthetic data"

**Response:** Study 2 results use no synthetic augmentation. Study 1 uses SMOTE only in the small-n imbalanced regime, with manifold validation (LID, Mahalanobis) reported in §3.6. We clarified in Limitations that SMOTE is appropriate when real-data balancing is infeasible, not as a substitute when adequate real samples exist.

**Manuscript changes:** Limitations §5.1; Abstract unchanged on this point (already distinguishes regimes).

---

## Anticipated Comment 8: Competing interests / patent

**Response:** The provisional application (63/847,316) was not converted and has lapsed. Competing Interests updated accordingly.

**Manuscript changes:** Competing Interests section.

---

## Checklist before uploading revision to JBI

- [ ] Paste **actual** reviewer comments above each response
- [ ] Run `imbalance_stress_test.py` and replace placeholder in §3.8
- [ ] Highlight changes in revised docx (if journal requires tracked changes)
- [ ] Verify all numbers against `supplementary/CANONICAL_RESULTS.md`
- [ ] Upload only via JBI revision portal when invited
