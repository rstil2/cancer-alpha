# JBI Manuscript Revision Workspace

**Purpose:** Prepare reviewer-response revisions **without altering the submitted manuscript** currently under review at *Journal of Biomedical Informatics*.

**Submitted snapshot date:** 2026-06-21  
**Author:** R. Craig Stillwell, PhD

---

## How to handle a manuscript that is already in review

### Do this now

1. **Do not upload anything from `working/` to the journal** until an editor explicitly requests a revision (minor/major revision letter).
2. **Treat `submitted_snapshot/` as read-only.** It is the frozen record of what JBI received.
3. **Prepare in `working/`** so you can respond quickly when reviewer comments arrive.
4. **Do not withdraw or resubmit** to another journal while JBI review is active (dual-submission violation).

### When JBI sends a decision

| Decision | What to do |
|----------|------------|
| **Minor / Major revision** | Use `working/Combined_Manuscript_JBI_REVISED.docx` as your starting point. Merge editor/reviewer requests into that file. Upload only through the journal revision portal when invited. |
| **Reject with encouragement to resubmit** | Ask whether resubmission should reference the prior manuscript ID. Use this folder as the resubmission base; update cover letter accordingly. |
| **Reject (final)** | You may submit the revised version elsewhere. Update competing-interests/patent language before any new submission. |
| **Accept** | No revision upload needed. Optionally align public GitHub/README with published text after proofs. |

### What you can safely change while under review

| Safe | Avoid |
|------|-------|
| Work in `science/jbi_revision/` | Replacing `science/Combined_Manuscript_JBI.docx` in place |
| Improve GitHub reproducibility (`src/pipeline/`, docs) | Public README claims that contradict the submitted PDF |
| Draft point-by-point responses | Submitting a new version to JBI without invitation |
| Run robustness analyses (imbalance test, etc.) | Dual submission to another journal |

---

## Folder layout

```
jbi_revision/
├── README.md                          ← this file
├── REVISION_CHANGELOG.md              ← every text/logic change vs submitted snapshot
├── apply_revisions.py                 ← regenerates working/ docx from snapshot
├── submitted_snapshot/                ← FROZEN copy of what was submitted
│   ├── Combined_Manuscript_JBI_SUBMITTED_2026-06-21.docx
│   ├── Combined_Manuscript_JBI.pdf
│   └── CoverLetter_JBI.docx
├── working/                           ← editable revision draft (not submitted)
│   ├── Combined_Manuscript_JBI_REVISED.docx
│   └── Response_to_Reviewers_ANTICIPATED.md
└── supplementary/
    ├── CANONICAL_RESULTS.md           ← single source of truth for all numbers
    └── REPRODUCTION_GUIDE.md          ← how to regenerate Study 2 from code
```

---

## Regenerate the revised manuscript

```bash
cd /Users/stillwell/projects/cancer-alpha/science/jbi_revision
../.venv/bin/python apply_revisions.py
```

Output: `working/Combined_Manuscript_JBI_REVISED.docx`

---

## Summary of proactive fixes (vs submitted version)

See `REVISION_CHANGELOG.md` for the full list. Highlights:

1. Title typo: Multi-**Model** → Multi-**Modal**
2. Softened overclaims (`clinical-grade`, `near-perfect`, `approaches perfect`)
3. Fixed AI disclosure (JAIR → JBI / Elsevier)
4. Updated competing interests (provisional patent lapsed)
5. Added **Ethics Statement** and **Author Contributions**
6. Expanded **Limitations** (external validation, balanced subsampling, Study 1/2 cohort differences)
7. Clarified **Data and Code Availability** with canonical pipeline path
8. Fixed biomarker typo (KLK4 → KLK3)
9. Added Study 2 **imbalance robustness** subsection (Methods placeholder + Limitations note)

---

## Next steps when reviewers respond

1. Paste reviewer comments into `working/Response_to_Reviewers_ANTICIPATED.md` (or create `Response_to_Reviewers_FINAL.md`).
2. Map each comment to a section in `REVISION_CHANGELOG.md`.
3. Edit `Combined_Manuscript_JBI_REVISED.docx` (or re-run `apply_revisions.py` after updating replacement rules).
4. Run `src/pipeline/step4_train_evaluate.py` if reproducibility is questioned.
5. Submit revision package only via JBI's invited revision workflow.

---

## Contact

R. Craig Stillwell — craig.stillwell@gmail.com
