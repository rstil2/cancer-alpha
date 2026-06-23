# Experiments

Robustness and supplementary analyses for the JBI manuscript.

| Script | Purpose | Results |
|--------|---------|---------|
| [`imbalance_stress_test.py`](imbalance_stress_test.py) | Train balanced → test natural prevalence | [results JSON](../science/jbi_revision/supplementary/imbalance_stress_test_results.json) |
| [`negative_control_biology_test.py`](negative_control_biology_test.py) | Shuffle pathway labels (planned) | — |
| [`cup_validation.py`](cup_validation.py) | Historical CUP simulation (SciReports era) | Not in JBI public claims |
| [`subtype_prediction.py`](subtype_prediction.py) | Subtype classification | SciReports supplement |

Run imbalance test (requires `data/real_tcga_large/` locally):

```bash
python experiments/imbalance_stress_test.py
```
