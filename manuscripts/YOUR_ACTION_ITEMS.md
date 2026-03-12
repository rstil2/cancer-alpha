# YOUR ACTION ITEMS - Quick Checklist

**Read This First**: I've applied all the critical corrections to your manuscript. Here's exactly what YOU need to do now.

---

## 🔴 CRITICAL (Must Do Before Submission)

### 1. Update Patent Filing Date ⚠️ IMPORTANT!

**File**: `Oncura_CORRECTED_Ready_for_Submission.md`

**Action**:
1. Open the file
2. Search for: `filed March 2024`
3. Replace with your **actual patent filing date**
4. Delete the line that says: `**Note to author**: Please replace...`

**Why**: "March 2024" is a placeholder. Journals require the actual filing date.

---

### 2. Transfer Corrections to Your Word Document

**You Have Two Options**:

**Option A - Manual Copy/Paste** (Recommended, 30 min)
1. Open your Word doc: `Oncura_Revised_Manuscript_FINAL_with_Figures-THIS IS IT.docx`
2. Open corrected markdown: `Oncura_CORRECTED_Ready_for_Submission.md`
3. Copy these 5 sections from markdown to Word:
   - Author Contributions (after Acknowledgments)
   - Enhanced Data and Code Availability (replace old version)
   - Enhanced Competing Interests (replace old version, **with YOUR patent date**)
   - Ethics Approval (after Funding, before References)
   - All corrected table numbers (Tables 3-12)

**Option B - Convert & Merge** (If you know pandoc)
1. Convert markdown to Word: `pandoc Oncura_CORRECTED_Ready_for_Submission.md -o temp.docx`
2. Copy sections from temp.docx to your final document
3. Verify figures transferred correctly

---

### 3. Verify Figures Display in Word

**Action**:
1. Open your Word document
2. Scroll to each figure location
3. Confirm you see all 4 figures clearly:
   - Figure 1: Model Performance Comparison
   - Figure 2: Cancer Type-Specific Performance
   - Figure 3: Benchmarking Against State-of-the-Art
   - Figure 4: Feature Importance and SHAP Analysis

**If figures are missing**: The figure files exist in `manuscript_figures/` - you may need to re-embed them.

---

## 🟡 RECOMMENDED (Before Submission)

### 4. Choose Your Target Journal

**Options** (from SUBMISSION_READY.md):
- **Bioinformatics** (IF: 5.8) - Excellent fit for methods
- **Genome Biology** (IF: 12.3) - Excellent fit for genomics
- **Nature Communications** (IF: 16.6) - High impact option
- **BMC Bioinformatics** (IF: 3.2) - Open access option

**Action**: Pick one journal → check their submission requirements

---

### 5. Final Proofread

**Quick Checks**:
- [ ] Patent date is **your actual date** (not "March 2024")
- [ ] All 4 figures display correctly
- [ ] All tables numbered 1-12 (no duplicates)
- [ ] Author Contributions section present
- [ ] Ethics Statement present
- [ ] Enhanced Data Availability present
- [ ] Run spell-check one more time

---

## 📋 SIMPLIFIED 3-STEP PROCESS

If you're short on time, do this minimum:

1. **Fix patent date** (2 minutes)
   - Open `Oncura_CORRECTED_Ready_for_Submission.md`
   - Replace "March 2024" with actual date
   
2. **Update Word doc** (30 minutes)
   - Copy 5 corrected sections into your .docx
   - Use CORRECTIONS_APPLIED_SUMMARY.md as your guide
   
3. **Verify & Submit** (10 minutes)
   - Check figures display
   - Save as final version
   - Submit to journal with reproducibility package

**Total time: ~45 minutes**

---

## 📁 FILES YOU NEED

All files are in: `/Users/stillwell/projects/cancer-alpha/manuscripts/`

**Main Files**:
- `Oncura_CORRECTED_Ready_for_Submission.md` ← Your corrected manuscript
- `Oncura_Revised_Manuscript_FINAL_with_Figures-THIS IS IT.docx` ← Your Word doc to update

**Helper Files**:
- `CORRECTIONS_APPLIED_SUMMARY.md` ← Detailed summary of what was fixed
- `QUICK_FIX_TEXT.md` ← Copy-paste text for each correction
- `YOUR_ACTION_ITEMS.md` ← This file

**Submission Package**:
- `Oncura_Reproducibility_Package_Final.zip` (6 MB) ← Ready to submit with manuscript

---

## ✅ WHEN YOU'RE DONE

Your final submission should include:

1. **Main Manuscript** (Word document with all corrections)
   - Author Contributions ✓
   - Enhanced Data Availability ✓
   - Enhanced Competing Interests (with YOUR patent date) ✓
   - Ethics Statement ✓
   - Correct table numbering (1-12) ✓
   - All 4 figures embedded ✓

2. **Reproducibility Package** 
   - `Oncura_Reproducibility_Package_Final.zip` (already ready)

3. **Cover Letter**
   - Use template from SUBMISSION_READY.md
   - Customize for your chosen journal

---

## 🆘 IF YOU GET STUCK

**Question**: Which markdown file is the corrected one?
**Answer**: `Oncura_CORRECTED_Ready_for_Submission.md`

**Question**: Where do I find the exact text to add?
**Answer**: Open `CORRECTIONS_APPLIED_SUMMARY.md` - it shows exactly what was added where

**Question**: Do I need to edit the Word doc or the markdown?
**Answer**: Edit your Word doc using the corrected markdown as reference

**Question**: What about the patent filing date?
**Answer**: You MUST replace "March 2024" with your actual filing date - this is critical!

---

## 🎯 YOUR GOAL

**Transform This**: Manuscript with 5 critical issues  
**Into This**: Submission-ready manuscript in ~45 minutes  
**Result**: Journal submission TODAY ✨

You've got this! All the hard work is done - just these final touches.

---

**Created**: February 1, 2026  
**Priority**: HIGH - Do this before submitting  
**Time Required**: 45-60 minutes  
**Difficulty**: Easy (mostly copy/paste)
