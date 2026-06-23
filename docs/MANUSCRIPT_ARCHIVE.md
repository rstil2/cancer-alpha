# Manuscript archive index

**Do not edit historical files for JBI submission.** The submitted version is frozen in `science/jbi_revision/submitted_snapshot/`.

## Active (use these)

| File | Status |
|------|--------|
| `science/Combined_Manuscript_JBI.docx` | Submitted to JBI (local; `science/` gitignored) |
| `science/jbi_revision/working/Combined_Manuscript_JBI_REVISED.docx` | Reviewer-prep draft |
| `science/jbi_revision/submitted_snapshot/` | Frozen submit copy |
| `preprints/` | bioRxiv materials |

## Historical — Nature / Science era

- `science/Nature_Manuscript_*.docx`, `NatureComms_*`
- `science/FINAL_SUBMISSION_PACKAGE.md`, `NATURE_SUBMISSION_SUMMARY.md`
- `manuscripts/Oncura_*Manuscript*.md` with 96.5%, HIPAA, production-ready claims

## Historical — Scientific Reports era

- `manuscripts/Combined_Manuscript_SciReports_2026.md`
- `manuscripts/Response_to_Editor_SciReports_2026.md`
- `manuscripts/Cover_Letter_SciReports_*.md`
- Contains **97.6% CUP** claims (simulated held-out TCGA, not independent CUP cohorts)

## Historical — JAIR era

- `science/jair_submission/`
- `science/Combined_Manuscript_JAIR.docx`

## Historical — AIM / production system framing

- `manuscripts/Oncura_Complete_Revised_Manuscript_AIM.md`
- `manuscripts/Oncura_Updated_Manuscript_2025.md`
- Claims: Epic/Cerner, HIPAA, 99.97% uptime, 96.5% accuracy

## Utility scripts (safe to use)

| Script | Purpose |
|--------|---------|
| `manuscripts/validate_manuscript_references.py` | Pre-commit reference check |
| `manuscripts/tools/apply_reviewer_revisions.py` | Docx revision helper |
| `manuscripts/tools/expand_manuscript.py` | Section expansion |
| `manuscripts/tools/format_for_bib.py` | Bibliography formatting |

## Rules for new edits

1. Only edit files under `science/jbi_revision/working/` until JBI invites revision.
2. Cross-check all numbers against [docs/CANONICAL.md](../docs/CANONICAL.md).
3. Never reintroduce: 96.5%, 97.6% CUP (public), 4,913 samples, "clinical-grade" without qualification.
