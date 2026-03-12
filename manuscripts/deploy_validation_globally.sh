#!/bin/bash
#
# Global Manuscript Reference Validation Deployment Script
#
# This script deploys the manuscript reference validation system to all
# manuscript directories across your system.
#
# Usage:
#   ./deploy_validation_globally.sh
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VALIDATOR_SCRIPT="$SCRIPT_DIR/validate_manuscript_references.py"
PRE_COMMIT_HOOK="$SCRIPT_DIR/../.git/hooks/pre-commit"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}Manuscript Reference Validation - Global Deployment${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Check that validation script exists
if [ ! -f "$VALIDATOR_SCRIPT" ]; then
    echo "Error: Validation script not found at $VALIDATOR_SCRIPT"
    exit 1
fi

# List of manuscript directories to deploy to
MANUSCRIPT_DIRS=(
    "$HOME/Documents/Google Drive/Manuscripts/Gender Disparities in COVID-19 Vaccine Trials"
    "$HOME/Documents/Google Drive/Project 29 - Gender Disparities in Clincal Trials of Infectous Diseases II/data-gen/Manuscript files"
    "$HOME/Documents/Google Drive/Project 31 - Sex Equity in Clinical Trials/manuscript"
    "$HOME/Documents/Google Drive/Project 31 - Sex Equity in Clinical Trials/submission/manuscript"
    "$HOME/Documents/Google Drive/Project 33 - Bias in Reference Genomes/Submission_Package/Manuscript"
    "$HOME/Documents/Google Drive/Project 36 - Cancer Genomics/manuscripts"
    "$HOME/Documents/Google Drive/Project 36 - Cancer Genomics/manuscript_submission_package"
    "$HOME/Documents/GitHub/sex-equity-clinical-trials/manuscript"
    "$HOME/Documents/GitHub/sex-equity-clinical-trials/submission/manuscript"
    "$HOME/Documents/GitHub/sex-bias-genomes/Manuscript"
)

# Function to find git root for a directory
find_git_root() {
    local dir="$1"
    local current="$dir"
    
    while [ "$current" != "/" ]; do
        if [ -d "$current/.git" ]; then
            echo "$current"
            return 0
        fi
        current=$(dirname "$current")
    done
    
    return 1
}

# Function to deploy to a single directory
deploy_to_directory() {
    local target_dir="$1"
    
    if [ ! -d "$target_dir" ]; then
        echo -e "${YELLOW}⚠️  Directory not found: $target_dir${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Deploying to: $target_dir${NC}"
    
    # Copy validation script
    cp "$VALIDATOR_SCRIPT" "$target_dir/"
    chmod +x "$target_dir/validate_manuscript_references.py"
    echo "  ✓ Copied validation script"
    
    # Copy documentation
    if [ -f "$SCRIPT_DIR/MANUSCRIPT_REFERENCE_REQUIREMENTS.md" ]; then
        cp "$SCRIPT_DIR/MANUSCRIPT_REFERENCE_REQUIREMENTS.md" "$target_dir/"
        echo "  ✓ Copied requirements documentation"
    fi
    
    if [ -f "$SCRIPT_DIR/REFERENCE_VALIDATION_README.md" ]; then
        cp "$SCRIPT_DIR/REFERENCE_VALIDATION_README.md" "$target_dir/"
        echo "  ✓ Copied README"
    fi
    
    # Find git root and install pre-commit hook
    git_root=$(find_git_root "$target_dir")
    if [ $? -eq 0 ]; then
        hooks_dir="$git_root/.git/hooks"
        if [ -d "$hooks_dir" ]; then
            # Create custom pre-commit hook for this repo
            cat > "$hooks_dir/pre-commit" << 'EOF'
#!/bin/bash
#
# Pre-commit hook to validate manuscript references
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get the root directory of the git repository
REPO_ROOT=$(git rev-parse --show-toplevel)

# Find all manuscript directories in this repo
find_manuscript_dirs() {
    find "$REPO_ROOT" -type d \( -name "manuscript" -o -name "manuscripts" -o -name "Manuscript" \) 2>/dev/null
}

# Find validator script
find_validator() {
    local manuscript_dirs=$(find_manuscript_dirs)
    for dir in $manuscript_dirs; do
        if [ -f "$dir/validate_manuscript_references.py" ]; then
            echo "$dir/validate_manuscript_references.py"
            return 0
        fi
    done
    return 1
}

VALIDATOR=$(find_validator)

if [ -z "$VALIDATOR" ]; then
    # No validator found, skip validation
    exit 0
fi

# Get list of staged manuscript files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.md$|\.txt$' | grep -v README | grep -v TODO | grep -v LICENSE)

if [ -z "$STAGED_FILES" ]; then
    exit 0
fi

echo -e "${GREEN}Validating manuscript references...${NC}"

VALIDATION_FAILED=0
for FILE in $STAGED_FILES; do
    # Check if file is in a manuscript directory
    if [[ "$FILE" =~ [Mm]anuscript ]]; then
        FULL_PATH="$REPO_ROOT/$FILE"
        
        if [ -f "$FULL_PATH" ]; then
            echo "Checking: $FILE"
            
            python3 "$VALIDATOR" "$FULL_PATH" > /tmp/manuscript_validation.txt 2>&1
            
            if [ $? -ne 0 ]; then
                echo -e "${RED}❌ Reference validation failed for: $FILE${NC}"
                cat /tmp/manuscript_validation.txt
                VALIDATION_FAILED=1
            else
                echo -e "${GREEN}✅ References validated${NC}"
            fi
        fi
    fi
done

rm -f /tmp/manuscript_validation.txt

if [ $VALIDATION_FAILED -ne 0 ]; then
    echo ""
    echo -e "${RED}========================================================================${NC}"
    echo -e "${RED}COMMIT ABORTED: Manuscript reference validation failed${NC}"
    echo -e "${RED}========================================================================${NC}"
    echo ""
    echo -e "${YELLOW}To bypass (NOT RECOMMENDED): git commit --no-verify${NC}"
    echo ""
    exit 1
fi

exit 0
EOF
            chmod +x "$hooks_dir/pre-commit"
            echo "  ✓ Installed pre-commit hook in git repo: $git_root"
        fi
    else
        echo "  ⚠️  Not a git repository, skipping pre-commit hook"
    fi
    
    echo ""
}

# Deploy to all directories
echo "Deploying validation system to all manuscript directories..."
echo ""

DEPLOYED=0
FAILED=0

for dir in "${MANUSCRIPT_DIRS[@]}"; do
    if deploy_to_directory "$dir"; then
        ((DEPLOYED++))
    else
        ((FAILED++))
    fi
done

echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}Deployment Summary:${NC}"
echo -e "  ✓ Successfully deployed to: $DEPLOYED directories"
if [ $FAILED -gt 0 ]; then
    echo -e "${YELLOW}  ⚠️  Failed/skipped: $FAILED directories${NC}"
fi
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo "The manuscript reference validation system is now deployed globally."
echo "All git commits in these directories will automatically validate references."
echo ""
echo "Manual validation: python3 validate_manuscript_references.py <manuscript.md>"
echo ""
