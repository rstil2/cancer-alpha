# Manuscript Reference Requirements

## Overview

All manuscripts must include proper references. This is a mandatory requirement and will be automatically validated before commits.

## Requirements

### 1. References Section
- Every manuscript must have a clearly marked "References" or "## References" section
- The section must be at the end of the manuscript (or before supplementary materials)

### 2. Minimum Number of References
- **Minimum: 5 references** (can be adjusted based on manuscript type)
- Most full manuscripts should have 20-70 references depending on scope

### 3. Reference Format
Each reference must include:
- **Author names** (proper capitalization)
- **Publication year** (4-digit year: 1900-2099)
- **Complete citation information** (journal, volume, pages, DOI, etc.)
- **Minimum length of 30 characters**

### 4. No Placeholders
The following are **strictly prohibited**:
- `[citation needed]`
- `[ref]`, `[reference]`
- `[add reference]`
- `[TODO: reference]`, `[TODO: citation]`
- `[insert reference]`
- `[placeholder]`
- `et al. (unpublished)`
- `et al. (in preparation)`

## Validation

### Automatic Validation (Pre-commit Hook)
A Git pre-commit hook automatically validates all manuscript files before they are committed:
- Runs on all `.md` and `.txt` files in the `manuscripts/` directory
- Blocks commits if validation fails
- Can be bypassed with `git commit --no-verify` (NOT RECOMMENDED)

### Manual Validation
You can manually validate manuscripts using the validation script:

```bash
# Validate a single manuscript
python3 validate_manuscript_references.py my_manuscript.md

# Check all manuscripts in current directory
python3 validate_manuscript_references.py --check-all

# Customize minimum reference count
python3 validate_manuscript_references.py --min-refs 10 my_manuscript.md

# Get JSON output for scripting
python3 validate_manuscript_references.py --json my_manuscript.md
```

## Example Valid Reference Format

```markdown
## References

1. Vamathevan J, Clark D, Czodrowski P, et al. Applications of machine learning in drug discovery and development. Nat Rev Drug Discov. 2019;18(6):463-477.

2. Topol EJ. High-performance medicine: the convergence of human and artificial intelligence. Nat Med. 2019;25(1):44-56.

3. Hanahan D, Weinberg RA. Hallmarks of cancer: the next generation. Cell. 2011;144(5):646-674.

4. Sanchez-Vega F, Mina M, Armenia J, et al. Oncogenic signaling pathways in The Cancer Genome Atlas. Cell. 2018;173(2):321-337.

5. Sun D, Wang M, Li A. A multimodal deep neural network for human breast cancer prognosis prediction by integrating multi-dimensional data. IEEE/ACM Trans Comput Biol Bioinform. 2019;16(3):841-850.
```

## Common Issues and Solutions

### Issue: "No References section found"
**Solution:** Add a section header at the end of your manuscript:
```markdown
## References
```

### Issue: "Insufficient references"
**Solution:** Ensure you have at least 5 numbered references. For full research papers, aim for 20+ references covering:
- Background and context
- Related work and prior art
- Methods and techniques
- Validation and comparison

### Issue: "Reference X: No publication year found"
**Solution:** Every reference must include a 4-digit publication year:
```markdown
# Bad
1. Smith J. Cancer genomics. Nature.

# Good  
1. Smith J. Cancer genomics. Nature. 2021;543:123-134.
```

### Issue: "Placeholder citation found"
**Solution:** Replace all placeholder text with actual references:
```markdown
# Bad
Our results are consistent with prior work [citation needed].

# Good
Our results are consistent with prior work.^1
```

## Reference Management Best Practices

### 1. Use Real References Only
- Never use placeholder citations
- Always include complete bibliographic information
- Verify DOIs and URLs work

### 2. Cite Appropriately
- Primary sources for key claims
- Recent work (last 5 years) for active areas
- Seminal papers for established concepts
- Methods papers for techniques used

### 3. Format Consistently
- Use the same citation style throughout
- Common styles: Vancouver (numbered), APA, Harvard
- Match target journal requirements

### 4. Organize References
- Number references in order of appearance
- Group related references when appropriate
- Include supplementary references if needed

## Troubleshooting

### Validation Script Not Running
```bash
# Check if script is executable
ls -la validate_manuscript_references.py

# If not, make it executable
chmod +x validate_manuscript_references.py
```

### Pre-commit Hook Not Working
```bash
# Check hook permissions
ls -la ../.git/hooks/pre-commit

# Make executable if needed
chmod +x ../.git/hooks/pre-commit

# Test the hook manually
../.git/hooks/pre-commit
```

### Need to Commit Without Validation
In rare cases where you need to commit a draft manuscript:
```bash
# Use --no-verify flag (use sparingly)
git commit --no-verify -m "Draft manuscript - references to be added"
```

## Integration with AI Assistants

When working with AI assistants (like Warp AI):
1. The assistant has been instructed to always include references in manuscripts
2. The assistant will reference this document when creating manuscripts
3. Validation will catch any oversights before commit

## Contact

For issues or questions about reference validation:
- Check this document first
- Review the validation script: `validate_manuscript_references.py`
- Contact: craig.stillwell@gmail.com
