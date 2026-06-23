# Contributing to Oncura

Thank you for contributing. Start with [RESEARCH.md](RESEARCH.md) and [docs/CANONICAL.md](docs/CANONICAL.md) before changing numbers or claims.

## Legal

- **Academic use:** permitted under [LICENSE](LICENSE)
- **Patent:** provisional 63/847,316 filed 2024, **lapsed** — no patent in force ([PATENTS.md](PATENTS.md))
- **Commercial use:** contact craig.stillwell@gmail.com

## High-value contributions

- Reproducibility fixes for `src/pipeline/` and Study 1 (`src/pipeline_study1/`)
- Robustness experiments under `experiments/`
- Documentation aligned with the JBI manuscript
- Bug fixes in `cancer_genomics_ai_demo_minimal/` (demo only)

## Please avoid

- Reintroducing legacy metrics (96.5%, 97.6% CUP, competitive leaderboard claims)
- Marketing language in README or demo UI
- Editing frozen manuscript files in `science/jbi_revision/submitted_snapshot/`

## Pull requests

1. Cross-check metrics against `docs/CANONICAL.md`
2. Note whether changes affect demo vs research pipeline
3. Manuscript reference validation runs on pre-commit for `manuscripts/*.md`

## Contact

craig.stillwell@gmail.com — subject "Oncura Academic Collaboration"
