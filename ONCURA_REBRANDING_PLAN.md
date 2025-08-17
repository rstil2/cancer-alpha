# Oncura Rebranding Completion Plan

## ✅ **Completed Files:**
1. **README.md** - Main repository README fully updated to Oncura
2. **business/outreach_letter.md** - Complete rebranding to Oncura

## 🎯 **Critical Files Requiring Manual Update:**

### **High Priority (Business Critical):**
1. **PATENTS.md** - Update patent documentation with Oncura name
2. **CONTRIBUTING.md** - Update contribution guidelines
3. **CHANGELOG.md** - Update project changelog
4. **LICENSE** - Review for any project name references

### **Academic & Research Papers:**
5. **preprints/** folder - All research papers and manuscripts
6. **manuscripts/** folder - Academic manuscripts and cover letters
7. **Citation references** - Update bibtex citations

### **Documentation:**
8. **docs/** folder - All documentation files
9. **Demo READMEs** - Demo and installation guides
10. **Technical guides** - Hospital deployment guides

### **Code & Configuration:**
11. **Python files** with comments/headers
12. **Shell scripts** with banner comments
13. **Docker files** with metadata
14. **Requirements files** with headers

## 🔧 **Batch Update Strategy:**

### **Option 1: Use Command Line (Recommended for Speed)**
```bash
# Navigate to project root
cd /Users/stillwell/projects/cancer-alpha

# Find and replace in all markdown files
find . -name "*.md" -type f -exec sed -i '' 's/Oncura/Oncura/g' {} \;

# Find and replace in all Python files
find . -name "*.py" -type f -exec sed -i '' 's/Oncura/Oncura/g' {} \;

# Find and replace in all text files
find . -name "*.txt" -type f -exec sed -i '' 's/Oncura/Oncura/g' {} \;

# Find and replace in shell scripts
find . -name "*.sh" -type f -exec sed -i '' 's/Oncura/Oncura/g' {} \;
find . -name "*.bat" -type f -exec sed -i '' 's/Oncura/Oncura/g' {} \;

# Find and replace in HTML files
find . -name "*.html" -type f -exec sed -i '' 's/Oncura/Oncura/g' {} \;
```

### **Option 2: Manual Priority Files (Most Important)**
Focus on these files for immediate business use:

1. **PATENTS.md** - Legal documentation
2. **preprints/README.md** - Research publication info
3. **docs/MASTER_INSTALLATION_GUIDE.md** - Technical guide
4. **cancer_genomics_ai_demo_minimal/README.md** - Demo documentation
5. **EXECUTIVE_SUMMARY.md** - Business summary
6. **COMMERCIALIZATION_STRATEGY.md** - Business strategy

### **Option 3: Phased Approach**
**Phase 1 (Immediate):** Business & legal files
**Phase 2 (This week):** Documentation & guides  
**Phase 3 (As needed):** Code comments & academic papers

## 📋 **Files That May Need Special Attention:**

### **Citation Updates:**
- Update BibTeX entries from "Oncura" to "Oncura"
- Research paper titles may need to remain as published
- Consider adding note: "Now known as Oncura"

### **URLs and Links:**
- GitHub repository URLs will remain cancer-alpha
- Consider creating redirect or alias
- Update internal documentation links as needed

### **Patent References:**
- Legal patent filings may need to remain as originally filed
- Add note about current branding as Oncura

## 🚀 **Recommended Next Steps:**

1. **Run the batch update commands above** (5 minutes)
2. **Review critical business files manually** (30 minutes)
3. **Test demo functionality** to ensure no broken references (10 minutes)
4. **Update any hardcoded strings in Python code** as needed

## 📊 **Files Updated Count:**
- **README.md**: ✅ Complete
- **business/outreach_letter.md**: ✅ Complete  
- **Remaining ~50 files**: 🔄 Ready for batch update

## ⚠️ **Important Notes:**

1. **Backup First**: Consider creating a git branch before bulk changes
2. **Published Research**: Academic papers already published should reference both names
3. **Patent Documentation**: Legal filings may need to maintain original names
4. **URLs**: GitHub repository name will remain cancer-alpha for consistency
5. **Testing**: Run demo after changes to ensure functionality

## 🎯 **Success Criteria:**
- [ ] All customer-facing documentation uses "Oncura"
- [ ] Demo functionality preserved
- [ ] Legal/patent references appropriately handled
- [ ] Academic citations properly updated
- [ ] Business materials fully rebranded
