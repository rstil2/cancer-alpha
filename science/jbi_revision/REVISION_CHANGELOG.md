# Revision Changelog (vs submitted snapshot 2026-06-21)

All changes are in `working/Combined_Manuscript_JBI_REVISED.docx` only.  
The submitted files in `science/` and `submitted_snapshot/` are unchanged.

---

## Administrative / compliance

| Item | Submitted | Revised |
|------|-----------|---------|
| AI disclosure journal | "JAIR policy" | Elsevier / JBI policy |
| Competing interests | Provisional patent active | Provisional lapsed; no patent in force |
| Ethics statement | Missing | Added (TCGA/ICGC public de-identified data) |
| Author contributions | Missing | Added (single author) |
| Data availability | Generic GitHub link | Canonical `src/pipeline/` + reproduction guide |

---

## Language / claims (tone-down)

| Location | Change |
|----------|--------|
| Title | Multi-**Model** → Multi-**Modal** |
| Abstract conclusions | Removed "clinical-grade" and "approaches perfect classification" |
| §4.7 heading | "Near-Perfect" → "High-Accuracy" |
| Introduction | "clinical-grade performance" → "high performance in the small-n regime" |
| Discussion §5 | Removed "approaches Bayes error"; explicit 95.0% / 98.4% |
| Conclusion | "clinical-grade" → "high balanced accuracy" |
| Biomarker | KLK4 → KLK3 |

---

## Scientific transparency

| Item | Change |
|------|--------|
| §5.1 Limitations | Expanded: Study 2 is internal TCGA held-out only; balanced subsampling caveat; Study 1 vs 2 cohort differences; SMOTE scope; VC bounds approximate; CUP not validated |
| §3.8 (new) | Sensitivity to class imbalance — methods + pointer to `imbalance_stress_test.py` |

---

## Not changed in this revision pass (await reviewer letter)

- Figure files (unchanged from submission)
- Study 1 numerical results **in submitted PDF** (95.0%, 92.1% ICGC) — frozen in `submitted_snapshot/`
- Reference list
- Deep learning baseline numbers (MLP 93.6%, TabTransformer 91.6%)
- VC dimension theoretical section (may soften further if reviewers challenge)

---

## Supplementary / docs updated (2026-06-23, no MS file edits)

| File | Change |
|------|--------|
| `docs/CANONICAL.md` | Submitted vs reproduced Study 1 table; ICGC 25% partial |
| `supplementary/CANONICAL_RESULTS.md` | Section A (frozen MS) vs B (pipeline); revision guidance |
| `supplementary/REPRODUCTION_GUIDE.md` | Study 1 reproduction section added |

`submitted_snapshot/` and `science/Combined_Manuscript_JBI.docx` (if present at repo root) remain untouched.

---

## Planned follow-ups (when revision invited)

- [ ] Run `imbalance_stress_test.py` and insert actual results into §3.8 / new Results subsection
- [ ] Add supplementary table with TCGA sample UUIDs for Study 2
- [ ] **Update `working/Combined_Manuscript_JBI_REVISED.docx`** Study 1 numbers to reproduced values (or explicit limitation paragraph) — use `supplementary/CANONICAL_RESULTS.md` section B
- [ ] Re-run `step4b_benchmark_architectures.py` and refresh complexity–accuracy R² in revision
- [ ] Point-by-point response mapped to reviewer comments
- [ ] Optional: tone-down README.md to match canonical docs (after acceptance or if desk-rejected)

---

## Regenerate revised docx

```bash
cd science/jbi_revision && ../.venv/bin/python apply_revisions.py
```

Edit `REPLACEMENTS` in `apply_revisions.py` to add further find-replace rules, then re-run.
