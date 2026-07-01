# Oncura — Research entry point

**Start here** for manuscript-aligned numbers, reproduction, and what *not* to cite from this repo.

The public [README](README.md) leads with **Study 2** (98.4%, reproducible). This page holds the full two-study picture for the JBI manuscript.

| Question | Answer |
|----------|--------|
| What is the paper about? | Experimental design dominates model architecture in multi-modal TCGA cancer classification ([JBI manuscript PDF](science/Combined_Manuscript_JBI.pdf)) |
| What numbers are official? | [docs/CANONICAL.md](docs/CANONICAL.md) |
| How do I reproduce Study 2 (98.4%)? | [src/pipeline/](src/pipeline/) → [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md) |
| How do I reproduce Study 1? | `python src/pipeline_study1/run_all.py` → `data/study1_results/` |
| JBI revision prep (under review) | [science/jbi_revision/](science/jbi_revision/) |
| Interactive UI demo (not paper data) | [cancer_genomics_ai_demo_minimal/](cancer_genomics_ai_demo_minimal/) |

## Two studies — do not merge

| | Study 1 | Study 2 |
|---|---------|---------|
| n | 158 | 1,248 |
| Features | 110 | 4,063 |
| Balance | SMOTE | Real subsampling |
| Best | LightGBM+SMOTE **82.4%** CV reproduced ([MS submitted: 95.0%](docs/CANONICAL.md)) | LightGBM 98.4% (test) |
| External val | ICGC n=76 **25%** partial reproduced ([MS: 92.1%](docs/CANONICAL.md)) | TCGA held-out only |

## Repository zones

```
src/pipeline/           ← canonical Study 2 (paper)
src/pipeline_study1/    ← Study 1 script map
experiments/            ← robustness tests (imbalance, etc.)
cancer_genomics_ai_demo_minimal/  ← UI demo (~70%), NOT manuscript reproduction
archive/experiments/    ← legacy scaling/download scripts (ignore for papers)
archive/business/       ← historical commercial docs (not research claims)
docs/MANUSCRIPT_ARCHIVE.md  ← which manuscript files are historical
```

## Preprint

[bioRxiv 10.1101/2025.07.22.666135](https://www.biorxiv.org/content/10.1101/2025.07.22.666135v1)

## Citation

See [README.md](README.md#citation).
