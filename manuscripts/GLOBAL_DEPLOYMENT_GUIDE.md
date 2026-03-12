# Global Manuscript Reference Validation - Quick Reference

## System Status: ✅ ACTIVE ACROSS ALL MANUSCRIPT DIRECTORIES

The manuscript reference validation system has been deployed globally to prevent missing or placeholder references from being committed to any of your manuscript repositories.

## What Was Deployed

### Files in Each Manuscript Directory
- `validate_manuscript_references.py` - Validation script
- `MANUSCRIPT_REFERENCE_REQUIREMENTS.md` - Detailed requirements
- `REFERENCE_VALIDATION_README.md` - Quick-start guide

### Git Repositories with Pre-commit Hooks
The following repositories will automatically validate manuscripts on commit:
1. cancer-alpha
2. Gender Disparities in COVID-19 Vaccine Trials  
3. data-gen (Project 29)
4. sex-equity-clinical-trials
5. sex-bias-genomes

## All Manuscript Directories Protected

✅ **10 directories** now have validation:

1. Cancer Alpha
2. Gender Disparities in COVID-19 Vaccine Trials
3. Project 29 - Gender Disparities in Clinical Trials of Infectious Diseases II
4. Project 31 - Sex Equity in Clinical Trials (main)
5. Project 31 - Sex Equity in Clinical Trials (submission)
6. Project 33 - Bias in Reference Genomes
7. Project 36 - Cancer Genomics (main)
8. Project 36 - Cancer Genomics (submission)
9. Sex Equity Clinical Trials (GitHub)
10. Sex Bias Genomes (GitHub)

## How It Works

### Automatic (Git Repositories)
When you commit manuscript files:
```bash
git add manuscript/my_paper.md
git commit -m "Update manuscript"
# ✓ Validation runs automatically
# ✓ Commit succeeds only if references are valid
```

### Manual Validation
From any manuscript directory:
```bash
# Validate a single manuscript
python3 validate_manuscript_references.py my_manuscript.md

# Check all manuscripts in directory
python3 validate_manuscript_references.py --check-all
```

## What Gets Checked

Every manuscript must have:
- ✅ A "References" section
- ✅ At least 5 references (configurable)
- ✅ Proper formatting (authors, years, complete citations)
- ✅ No placeholders like `[citation needed]`

## Common Scenarios

### Creating a New Manuscript
Just include a References section with at least 5 real references. The system will validate it when you commit.

### Editing Existing Manuscripts
The validation runs automatically. If you add citations, make sure to add the references.

### Working Outside Git
Use manual validation:
```bash
python3 validate_manuscript_references.py manuscript.md
```

### Emergency Commit (Draft Work)
If you absolutely must commit without validation:
```bash
git commit --no-verify -m "WIP: Draft only"
```
**WARNING:** Only use for drafts. Never submit without proper references.

## Redeployment

### If You Create a New Manuscript Directory
Run the global deployment script:
```bash
cd /Users/stillwell/projects/cancer-alpha/manuscripts
./deploy_validation_globally.sh
```

### Or Deploy Manually to One Directory
```bash
cd /Users/stillwell/projects/cancer-alpha/manuscripts
cp validate_manuscript_references.py <new_manuscript_dir>/
cp MANUSCRIPT_REFERENCE_REQUIREMENTS.md <new_manuscript_dir>/
cp REFERENCE_VALIDATION_README.md <new_manuscript_dir>/
chmod +x <new_manuscript_dir>/validate_manuscript_references.py
```

## Troubleshooting

### "No References section found"
Add a References section:
```markdown
## References

1. Author A. Title. Journal. 2020;12(3):45-67.
[... at least 5 total ...]
```

### "Insufficient references"
Add more references. Minimum is 5, but most papers need 20+.

### "Placeholder citation found"
Remove all `[citation needed]`, `[ref]`, etc. and replace with real references.

### Validation Not Running on Commit
Check if you're in a git repository:
```bash
git rev-parse --git-dir  # Should show .git
```

If not a git repo, use manual validation before submission.

## AI Assistant Integration

When working with AI assistants (Warp, ChatGPT, etc.):
- The system automatically enforces the rules
- AI assistants are aware of the requirements
- Validation catches any oversights

## Key Benefits

This system ensures:
- ❌ **Prevents** manuscripts without references from being committed
- ❌ **Blocks** placeholder citations from making it to journals  
- ✅ **Guarantees** every manuscript has proper references
- ✅ **Works** across all your projects automatically
- ✅ **Catches** issues before journal submission

## Documentation

Detailed documentation in each manuscript directory:
- `MANUSCRIPT_REFERENCE_REQUIREMENTS.md` - Complete requirements
- `REFERENCE_VALIDATION_README.md` - Usage guide
- `VALIDATION_SYSTEM_SUMMARY.md` - Implementation details (cancer-alpha only)

## Contact

For questions or issues:
- Review the documentation in any manuscript directory
- Examine the validation script source code
- Email: craig.stillwell@gmail.com

---

**Last Deployed:** February 3, 2026  
**Status:** ✅ ACTIVE AND PROTECTING ALL MANUSCRIPT DIRECTORIES
