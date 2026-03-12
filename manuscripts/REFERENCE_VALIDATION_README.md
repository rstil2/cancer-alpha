# Manuscript Reference Validation System

## Quick Start

This directory has an **automatic validation system** that ensures all manuscripts include proper references before they are committed to git.

### For Manual Validation

```bash
# Validate a single manuscript
python3 validate_manuscript_references.py my_manuscript.md

# Check all manuscripts
python3 validate_manuscript_references.py --check-all
```

### For Git Commits

The system automatically validates manuscripts when you commit:

```bash
# Normal commit (will auto-validate)
git add manuscripts/my_manuscript.md
git commit -m "Add manuscript"

# If validation fails, you'll see an error and the commit will be blocked
# Fix the issues and try again
```

## What Gets Validated

The system checks:
- ✅ Presence of a "References" section
- ✅ Minimum of 5 references (configurable)
- ✅ Proper formatting (author names, years, etc.)
- ✅ No placeholder citations like `[citation needed]`

## Files in This System

1. **validate_manuscript_references.py** - Main validation script
2. **MANUSCRIPT_REFERENCE_REQUIREMENTS.md** - Detailed requirements and guidelines
3. **../.git/hooks/pre-commit** - Git hook that runs validation automatically
4. **REFERENCE_VALIDATION_README.md** - This file

## Common Workflows

### Creating a New Manuscript

When creating a new manuscript, ensure you include a References section:

```markdown
# My New Manuscript

## Introduction
...

## Methods
...

## Results
...

## Discussion
...

## References

1. Author A, Author B. Title of paper. Journal Name. 2020;12(3):45-67.

2. Author C, Author D. Another paper. Journal Name. 2019;10(2):123-145.

[... at least 5 total references ...]
```

### Editing an Existing Manuscript

When editing, the validation will run automatically on commit. If you add citations in the text, make sure to add the corresponding references to the References section.

### Quick Validation Before Commit

Before committing, you can check your manuscript:

```bash
python3 validate_manuscript_references.py manuscripts/my_manuscript.md
```

## Troubleshooting

### Validation Fails on Commit

If you see an error like:
```
❌ COMMIT ABORTED: Manuscript reference validation failed
```

1. Check the error messages to see what's wrong
2. Run manual validation to see details: `python3 validate_manuscript_references.py <file>`
3. Fix the issues
4. Commit again

### Emergency: Need to Commit Without Validation

In rare cases (e.g., saving work in progress):

```bash
git commit --no-verify -m "WIP: Draft manuscript"
```

**WARNING:** Only use this for drafts. Never submit a manuscript without proper references.

### Script Not Found

If you get "command not found":

```bash
# Make sure you're in the manuscripts directory
cd /Users/stillwell/projects/cancer-alpha/manuscripts

# Make the script executable
chmod +x validate_manuscript_references.py

# Run with python3 explicitly
python3 validate_manuscript_references.py <manuscript>
```

## Integration with AI Tools

When working with AI assistants (Warp, ChatGPT, etc.):

1. Always remind them: "Include proper references - no placeholders"
2. The system will catch mistakes automatically
3. AI assistants should reference `MANUSCRIPT_REFERENCE_REQUIREMENTS.md`

## Customization

### Change Minimum Reference Count

```bash
# Require at least 10 references
python3 validate_manuscript_references.py --min-refs 10 my_manuscript.md
```

### Get Machine-Readable Output

```bash
# JSON output for scripts
python3 validate_manuscript_references.py --json my_manuscript.md
```

## Benefits

This system prevents:
- ❌ Submitting manuscripts without references
- ❌ Using placeholder citations
- ❌ Incomplete reference formatting
- ❌ Forgetting to add references

And ensures:
- ✅ Every manuscript has proper references
- ✅ References are formatted correctly
- ✅ No placeholder text makes it to submission
- ✅ Consistent quality across all manuscripts

## Support

For questions or issues:
1. Check `MANUSCRIPT_REFERENCE_REQUIREMENTS.md` for detailed guidance
2. Review the validation script: `validate_manuscript_references.py`
3. Contact: craig.stillwell@gmail.com
