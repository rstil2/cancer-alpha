# Quick Fix Text - Copy and Paste Directly into Manuscript

## 1. ETHICS STATEMENT
**Insert after "Funding" section, before "References"**

```
Ethics Approval and Consent to Participate

This study used publicly available de-identified genomic and clinical data from The Cancer Genome Atlas (TCGA) accessed through the Genomic Data Commons (GDC) portal under controlled-access authorization. All TCGA data were collected under protocols approved by institutional review boards at the original collection sites, with informed consent obtained from all participants as part of the TCGA Research Network. This secondary analysis of de-identified archival data does not constitute human subjects research under 45 CFR 46.102(l)(2) and does not require additional IRB approval per institutional policy at Campbellsville University.
```

---

## 2. AUTHOR CONTRIBUTIONS
**Insert after "Acknowledgments" section, before "Data and Code Availability"**

```
Author Contributions

R.C.S. conceived and designed the study, developed the novel AI methodological framework, performed all data processing and statistical analyses, implemented the production system, validated all results, and wrote the manuscript. As sole author, R.C.S. is responsible for all aspects of the work and approved the final version for submission.
```

---

## 3. ENHANCED DATA AVAILABILITY STATEMENT
**Replace existing "Data and Code Availability" section**

```
Data and Code Availability

Source Code and Processed Data: All code, processed feature matrices, and trained models are publicly available through Zenodo: Stillwell, R. C. (2025). Knowledge-Guided Multi-Modal Integration Improves Robustness and Accuracy in Multi-Cancer Genomic Classification [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17701703

The reproducibility package includes:
• Complete Python source code for all analyses
• Preprocessed feature matrices (1,200 samples × 2,000 features)
• Trained LightGBM models with all hyperparameters
• Complete analysis scripts reproducing all figures and tables
• Documentation and requirements files

Raw TCGA Genomic Data: Raw genomic data used in this study were accessed from The Cancer Genome Atlas (TCGA) through the Genomic Data Commons (GDC) Data Portal (https://portal.gdc.cancer.gov/) under controlled-access authorization. Researchers can obtain access by applying for dbGaP authorization (https://dbgap.ncbi.nlm.nih.gov/). Complete sample manifests with TCGA UUIDs are included in the DATA_PROVENANCE.md file in the Zenodo repository.

For questions regarding data access or reproducibility: craig.stillwell@gmail.com
```

---

## 4. ENHANCED COMPETING INTERESTS STATEMENT
**Replace existing "Competing Interests" section**

⚠️ **IMPORTANT**: Replace "March 15, 2024" with your ACTUAL patent filing date!

```
Competing Interests

The author has filed a provisional patent application (U.S. Provisional Patent Application No. 63/847,316, filed [INSERT ACTUAL FILING DATE]) covering the knowledge-guided multi-modal integration framework and biologically-validated interpretability methods described in this work. The provisional patent application is currently under examination. Academic and research use is freely permitted with proper attribution; commercial use or clinical deployment requires a separate licensing agreement. For licensing inquiries, contact craig.stillwell@gmail.com.
```

---

## 5. TABLE RENUMBERING
**Find and Replace these throughout the entire document:**

| Find This | Replace With |
|-----------|--------------|
| "Table 2: Comprehensive Ablation Study Results" | "Table 3: Comprehensive Ablation Study Results" |
| "Table 3: Single-Modality vs. Multi-Modal Performance" | "Table 4: Single-Modality vs. Multi-Modal Performance" |
| "Table 4: Feature Selection Approach Comparison" | "Table 5: Feature Selection Approach Comparison" |
| "Table 5: Balance Strategy Comparison" | "Table 6: Balance Strategy Comparison" |
| "Table 6: Per-Cancer-Type Ablation Impact" | "Table 7: Per-Cancer-Type Ablation Impact" |
| "Table 7: Direct Comparison on Our Dataset" | "Table 8: Direct Comparison on Our Dataset" |
| "Table 8: Model Performance Comparison" | "Table 9: Model Performance Comparison" |
| "Table 9: Cancer Type-Specific Performance" | "Table 10: Cancer Type-Specific Performance" |
| "Table 10: Academic Research Benchmarking" | "Table 12: Academic Research Benchmarking" |

**Note**: Table 11 (Robustness to Class Imbalance) is already correct - don't change it!

---

## VERIFICATION CHECKLIST

After making these changes, verify:

- [ ] Ethics statement appears before References section
- [ ] Author Contributions appears after Acknowledgments
- [ ] Data Availability section is expanded with full details
- [ ] Patent filing date is **YOUR ACTUAL DATE** (not placeholder!)
- [ ] All tables numbered 1-12 in sequential order
- [ ] No table number appears twice
- [ ] Table references in text match table numbers

---

## ESTIMATED TIME TO COMPLETE

- Adding 4 text sections: **15 minutes**
- Table renumbering (find/replace): **10 minutes**
- Verification: **5 minutes**

**Total: 30 minutes maximum**

Then you're ready to submit!
