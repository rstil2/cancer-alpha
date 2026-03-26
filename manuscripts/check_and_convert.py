#!/usr/bin/env python3
"""Check for pandoc/python-docx and convert markdown files to Word."""
import subprocess
import sys
import os

os.chdir("/Users/stillwell/projects/cancer-alpha/manuscripts")

# Check pandoc
r = subprocess.run(["which", "pandoc"], capture_output=True, text=True)
pandoc_path = r.stdout.strip()

# Check python-docx
try:
    import docx
    has_docx = True
except ImportError:
    has_docx = False

with open("/Users/stillwell/projects/cancer-alpha/manuscripts/convert_status.txt", "w") as f:
    f.write(f"pandoc: {pandoc_path or 'NOT FOUND'}\n")
    f.write(f"python-docx: {has_docx}\n")

    files = [
        "Combined_Manuscript_SciReports_2026.md",
        "Cover_Letter_SciReports_Revision_2026.md",
        "Response_to_Editor_SciReports_2026.md",
    ]

    if pandoc_path:
        for md in files:
            docx_name = md.replace(".md", ".docx")
            result = subprocess.run(
                ["pandoc", md, "-o", docx_name, "--from=markdown", "--to=docx"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                f.write(f"OK: {docx_name}\n")
            else:
                f.write(f"FAIL: {docx_name} - {result.stderr}\n")
    elif has_docx:
        f.write("Using python-docx fallback\n")
        # Will need a more complex conversion
        f.write("NEED_FALLBACK\n")
    else:
        f.write("NO CONVERTER AVAILABLE - need pandoc or python-docx\n")
        f.write("Install with: brew install pandoc\n")
        f.write("Or: pip install python-docx\n")

print("Done - check convert_status.txt")
