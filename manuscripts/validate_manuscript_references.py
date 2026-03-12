#!/usr/bin/env python3
"""
Manuscript Reference Validator

This script validates that manuscripts contain proper references before submission.
It checks for:
1. Presence of a References section
2. Minimum number of references (default: 5)
3. Proper reference formatting
4. No placeholder references

Usage:
    python validate_manuscript_references.py <manuscript_file>
    python validate_manuscript_references.py --check-all
"""

import sys
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import json

# Minimum number of references required
MIN_REFERENCES = 5

# Placeholder patterns to detect (case-insensitive)
PLACEHOLDER_PATTERNS = [
    r'\[citation needed\]',
    r'\[ref\]',
    r'\[reference\]',
    r'\[add reference\]',
    r'\[todo:?\s*reference\]',
    r'\[todo:?\s*citation\]',
    r'\[insert reference\]',
    r'\[placeholder\]',
    r'et al\.\s*\[?\d*\]?\s*\(unpublished\)',
    r'et al\.\s*\[?\d*\]?\s*\(in preparation\)',
]


def find_references_section(content: str) -> Tuple[bool, int, int]:
    """
    Find the References section in the manuscript.
    
    Returns:
        (has_section, start_line, end_line)
    """
    lines = content.split('\n')
    
    # Look for References header
    ref_patterns = [
        r'^#{1,3}\s*References?\s*$',
        r'^References?\s*$',
        r'^REFERENCES?\s*$',
    ]
    
    start_line = -1
    for i, line in enumerate(lines):
        for pattern in ref_patterns:
            if re.match(pattern, line.strip()):
                start_line = i
                break
        if start_line >= 0:
            break
    
    if start_line < 0:
        return (False, -1, -1)
    
    # Find end of references (next major section or end of file)
    end_line = len(lines)
    for i in range(start_line + 1, len(lines)):
        if re.match(r'^#{1,3}\s*\w+', lines[i]) or re.match(r'^[A-Z][A-Z\s]+$', lines[i].strip()):
            end_line = i
            break
    
    return (True, start_line, end_line)


def extract_references(content: str, start_line: int, end_line: int) -> List[str]:
    """Extract individual references from the References section."""
    lines = content.split('\n')[start_line:end_line]
    references = []
    
    # Match numbered references like "1. Author et al..."
    ref_pattern = r'^\d+\.\s+.+'
    
    for line in lines:
        stripped = line.strip()
        if re.match(ref_pattern, stripped):
            references.append(stripped)
    
    return references


def check_reference_format(reference: str) -> Tuple[bool, str]:
    """
    Check if a reference has proper formatting.
    
    Returns:
        (is_valid, error_message)
    """
    # Check minimum length
    if len(reference) < 30:
        return (False, "Reference too short (likely incomplete)")
    
    # Check for author names (at least one capital letter followed by lowercase)
    if not re.search(r'[A-Z][a-z]+', reference):
        return (False, "No author names detected")
    
    # Check for year (4 digits)
    if not re.search(r'\b(19|20)\d{2}\b', reference):
        return (False, "No publication year found")
    
    # Check for placeholder patterns
    for pattern in PLACEHOLDER_PATTERNS:
        if re.search(pattern, reference, re.IGNORECASE):
            return (False, f"Contains placeholder text: {pattern}")
    
    return (True, "")


def check_for_inline_placeholders(content: str) -> List[Tuple[int, str]]:
    """Find placeholder citations in the main text."""
    lines = content.split('\n')
    placeholders = []
    
    for i, line in enumerate(lines, 1):
        for pattern in PLACEHOLDER_PATTERNS:
            matches = re.finditer(pattern, line, re.IGNORECASE)
            for match in matches:
                placeholders.append((i, match.group()))
    
    return placeholders


def validate_manuscript(filepath: Path, min_refs: int = MIN_REFERENCES) -> Dict:
    """
    Validate a manuscript file for proper references.
    
    Args:
        filepath: Path to manuscript file
        min_refs: Minimum number of references required
    
    Returns:
        Dictionary with validation results
    """
    result = {
        'file': str(filepath),
        'valid': False,
        'errors': [],
        'warnings': [],
        'reference_count': 0
    }
    
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        result['errors'].append(f"Could not read file: {e}")
        return result
    
    # Check for References section
    has_section, start_line, end_line = find_references_section(content)
    
    if not has_section:
        result['errors'].append("No References section found")
        return result
    
    # Extract and count references
    references = extract_references(content, start_line, end_line)
    result['reference_count'] = len(references)
    
    if len(references) < min_refs:
        result['errors'].append(
            f"Insufficient references: {len(references)} found, "
            f"minimum {min_refs} required"
        )
    
    # Check each reference
    for i, ref in enumerate(references, 1):
        is_valid, error_msg = check_reference_format(ref)
        if not is_valid:
            result['errors'].append(f"Reference {i}: {error_msg}")
    
    # Check for inline placeholders
    inline_placeholders = check_for_inline_placeholders(content)
    if inline_placeholders:
        for line_num, placeholder in inline_placeholders:
            result['errors'].append(
                f"Line {line_num}: Placeholder citation found: {placeholder}"
            )
    
    # Determine if valid
    result['valid'] = len(result['errors']) == 0
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Validate manuscript references'
    )
    parser.add_argument(
        'manuscript',
        nargs='?',
        help='Path to manuscript file to validate'
    )
    parser.add_argument(
        '--check-all',
        action='store_true',
        help='Check all markdown/text manuscripts in the current directory'
    )
    parser.add_argument(
        '--min-refs',
        type=int,
        default=MIN_REFERENCES,
        help=f'Minimum number of references required (default: {MIN_REFERENCES})'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    # Determine which files to check
    files_to_check = []
    
    if args.check_all:
        manuscripts_dir = Path.cwd()
        files_to_check = list(manuscripts_dir.glob('*.md'))
        files_to_check.extend(manuscripts_dir.glob('*.txt'))
        # Filter out README and other common non-manuscript files
        files_to_check = [
            f for f in files_to_check 
            if not any(x in f.name.upper() for x in ['README', 'TODO', 'LICENSE'])
        ]
    elif args.manuscript:
        files_to_check = [Path(args.manuscript)]
    else:
        parser.print_help()
        return 1
    
    # Validate each file
    results = []
    for filepath in files_to_check:
        if not filepath.exists():
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            continue
        
        result = validate_manuscript(filepath, min_refs=args.min_refs)
        results.append(result)
    
    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        all_valid = True
        for result in results:
            print(f"\n{'='*80}")
            print(f"File: {result['file']}")
            print(f"{'='*80}")
            
            if result['valid']:
                print(f"✅ VALID - {result['reference_count']} references found")
            else:
                print(f"❌ INVALID")
                all_valid = False
            
            if result['errors']:
                print(f"\nErrors ({len(result['errors'])}):")
                for error in result['errors']:
                    print(f"  ❌ {error}")
            
            if result['warnings']:
                print(f"\nWarnings ({len(result['warnings'])}):")
                for warning in result['warnings']:
                    print(f"  ⚠️  {warning}")
        
        print(f"\n{'='*80}")
        if all_valid:
            print("✅ All manuscripts validated successfully")
            return 0
        else:
            print("❌ Some manuscripts have issues")
            return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
