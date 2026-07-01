#!/usr/bin/env python3
"""
COMPREHENSIVE DATA AUDIT SCRIPT
===============================
Identifies synthetic vs real TCGA data in violation of zero-synthetic policy

This script performs forensic analysis to detect:
- Synthetic/generated data patterns
- Invalid TCGA barcodes
- Identical/duplicated rows
- Statistical anomalies indicating artificial generation
- File integrity issues

Author: Cancer Alpha Project
Purpose: Cleanup synthetic data contamination
Rule: ZERO synthetic data allowed per user policy
"""

import os
import pandas as pd
import numpy as np
import json
import re
import hashlib
import sqlite3
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class TCGAAudit:
    def __init__(self, data_directory="/Users/stillwell/projects/cancer-alpha/data"):
        self.data_dir = Path(data_directory)
        self.audit_results = []
        self.tcga_barcode_pattern = re.compile(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}')
        self.red_flags = {
            'identical_rows': 0,
            'zero_variance_columns': 0,
            'invalid_barcodes': 0,
            'synthetic_patterns': 0,
            'suspicious_statistics': 0
        }
        
        print("🔍 TCGA Data Audit Starting...")
        print("=" * 60)
        print(f"📂 Scanning directory: {self.data_dir}")
        print(f"🚫 ZERO synthetic data policy enforced")
        print("=" * 60)
    
    def is_valid_tcga_barcode(self, sample_id):
        """Validate TCGA barcode format"""
        if not isinstance(sample_id, str):
            return False
        return bool(self.tcga_barcode_pattern.match(sample_id))
    
    def detect_synthetic_patterns(self, df, file_path):
        """Detect patterns indicating synthetic/generated data"""
        red_flags = []
        synthetic_score = 0
        
        # Check for identical rows
        if len(df) > 1:
            duplicate_mask = df.duplicated()
            duplicate_count = duplicate_mask.sum()
            if duplicate_count > len(df) * 0.1:  # >10% duplicates
                red_flags.append(f"High duplicate rate: {duplicate_count}/{len(df)} ({duplicate_count/len(df)*100:.1f}%)")
                synthetic_score += 3
        
        # Check for identical values across rows (major red flag from our analysis)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                unique_vals = df[col].nunique()
                if unique_vals == 1 and len(df) > 100:  # Same value for 100+ rows
                    red_flags.append(f"Column '{col}' has identical values across {len(df)} rows")
                    synthetic_score += 5
                elif unique_vals < len(df) * 0.01:  # <1% unique values
                    red_flags.append(f"Column '{col}' has suspiciously low variance: {unique_vals} unique values")
                    synthetic_score += 2
        
        # Check for artificial patterns in sample IDs
        if 'sample_id' in df.columns:
            invalid_barcodes = 0
            for sample_id in df['sample_id'].dropna():
                if not self.is_valid_tcga_barcode(sample_id):
                    invalid_barcodes += 1
            
            if invalid_barcodes > 0:
                red_flags.append(f"Invalid TCGA barcodes: {invalid_barcodes}/{len(df)}")
                synthetic_score += invalid_barcodes * 0.1
        
        # Check for repeated statistical values (like we saw)
        for col in numeric_cols:
            if col.endswith(('_mean', '_std', '_median', '_min', '_max')):
                if len(df) > 10 and df[col].nunique() == 1:
                    red_flags.append(f"Statistical column '{col}' has identical values - indicates artificial generation")
                    synthetic_score += 4
        
        # Determine authenticity
        if synthetic_score >= 10:
            authenticity = "SYNTHETIC"
        elif synthetic_score >= 5:
            authenticity = "SUSPICIOUS"
        elif len(red_flags) > 0:
            authenticity = "WARNING"
        else:
            authenticity = "AUTHENTIC"
        
        return authenticity, red_flags, synthetic_score
    
    def audit_csv_file(self, csv_path):
        """Audit a single CSV file"""
        try:
            print(f"📄 Auditing: {csv_path}")
            
            # Basic file info
            file_size = os.path.getsize(csv_path)
            
            # Read and analyze
            df = pd.read_csv(csv_path)
            n_samples, n_features = df.shape
            
            # Detect synthetic patterns
            authenticity, red_flags, synthetic_score = self.detect_synthetic_patterns(df, csv_path)
            
            # Compile audit result
            result = {
                'path': str(csv_path),
                'file_size_mb': file_size / (1024*1024),
                'n_samples': n_samples,
                'n_features': n_features,
                'authenticity_flag': authenticity,
                'synthetic_score': synthetic_score,
                'red_flags': '; '.join(red_flags) if red_flags else 'None',
                'reason': f"Synthetic score: {synthetic_score}. Issues: {len(red_flags)}"
            }
            
            # Print immediate findings
            if authenticity in ['SYNTHETIC', 'SUSPICIOUS']:
                print(f"🚨 {authenticity}: {csv_path}")
                print(f"   📊 Size: {n_samples:,} samples × {n_features} features")
                print(f"   ⚠️  Issues: {len(red_flags)}")
                for flag in red_flags[:3]:  # Show first 3 issues
                    print(f"      • {flag}")
                if len(red_flags) > 3:
                    print(f"      • ... and {len(red_flags)-3} more issues")
            else:
                print(f"✅ {authenticity}: {csv_path} ({n_samples:,} samples)")
            
            return result
            
        except Exception as e:
            print(f"❌ Error auditing {csv_path}: {e}")
            return {
                'path': str(csv_path),
                'file_size_mb': 0,
                'n_samples': 0,
                'n_features': 0,
                'authenticity_flag': 'ERROR',
                'synthetic_score': 0,
                'red_flags': f'Read error: {str(e)}',
                'reason': f'Failed to read: {str(e)}'
            }
    
    def run_comprehensive_audit(self):
        """Run audit on all CSV files in data directory"""
        print("\n🔍 COMPREHENSIVE DATA AUDIT")
        print("=" * 60)
        
        # Find all CSV files
        csv_files = list(self.data_dir.glob('**/*.csv'))
        print(f"📁 Found {len(csv_files)} CSV files to audit")
        
        if not csv_files:
            print("⚠️  No CSV files found!")
            return
        
        print("\n📊 DETAILED AUDIT RESULTS:")
        print("-" * 60)
        
        # Audit each file
        for csv_file in csv_files:
            result = self.audit_csv_file(csv_file)
            self.audit_results.append(result)
        
        # Generate summary
        self.generate_audit_summary()
        self.save_audit_report()
        
        return self.audit_results
    
    def generate_audit_summary(self):
        """Generate and print audit summary"""
        if not self.audit_results:
            print("No audit results to summarize")
            return
        
        # Count by authenticity
        authenticity_counts = Counter([r['authenticity_flag'] for r in self.audit_results])
        total_samples = sum([r['n_samples'] for r in self.audit_results])
        synthetic_samples = sum([r['n_samples'] for r in self.audit_results if r['authenticity_flag'] in ['SYNTHETIC', 'SUSPICIOUS']])
        
        print(f"\n" + "=" * 60)
        print("📋 AUDIT SUMMARY")
        print("=" * 60)
        print(f"📄 Total files audited: {len(self.audit_results)}")
        print(f"📊 Total samples found: {total_samples:,}")
        print(f"🚨 Synthetic/suspicious samples: {synthetic_samples:,} ({synthetic_samples/total_samples*100:.1f}%)")
        
        print(f"\n🔍 AUTHENTICITY BREAKDOWN:")
        for auth_type, count in authenticity_counts.items():
            samples_of_type = sum([r['n_samples'] for r in self.audit_results if r['authenticity_flag'] == auth_type])
            print(f"   {auth_type:>12}: {count:>3} files ({samples_of_type:>8,} samples)")
        
        # Critical findings
        synthetic_files = [r for r in self.audit_results if r['authenticity_flag'] == 'SYNTHETIC']
        if synthetic_files:
            print(f"\n🚨 CRITICAL: {len(synthetic_files)} files flagged as SYNTHETIC:")
            for file_result in synthetic_files:
                print(f"   • {file_result['path']} ({file_result['n_samples']:,} samples)")
        
        # High-priority suspicious files
        suspicious_files = [r for r in self.audit_results if r['authenticity_flag'] == 'SUSPICIOUS']
        if suspicious_files:
            print(f"\n⚠️  WARNING: {len(suspicious_files)} files flagged as SUSPICIOUS:")
            for file_result in suspicious_files[:5]:  # Show first 5
                print(f"   • {file_result['path']} ({file_result['n_samples']:,} samples)")
            if len(suspicious_files) > 5:
                print(f"   • ... and {len(suspicious_files)-5} more suspicious files")
    
    def save_audit_report(self):
        """Save detailed audit report"""
        # Save CSV report
        audit_df = pd.DataFrame(self.audit_results)
        report_path = self.data_dir.parent / "data_audit_report.csv"
        audit_df.to_csv(report_path, index=False)
        
        # Save JSON summary
        summary = {
            'audit_timestamp': pd.Timestamp.now().isoformat(),
            'total_files': len(self.audit_results),
            'total_samples': sum([r['n_samples'] for r in self.audit_results]),
            'synthetic_files': len([r for r in self.audit_results if r['authenticity_flag'] == 'SYNTHETIC']),
            'synthetic_samples': sum([r['n_samples'] for r in self.audit_results if r['authenticity_flag'] in ['SYNTHETIC', 'SUSPICIOUS']]),
            'authenticity_breakdown': dict(Counter([r['authenticity_flag'] for r in self.audit_results])),
            'files_to_purge': [r['path'] for r in self.audit_results if r['authenticity_flag'] in ['SYNTHETIC', 'SUSPICIOUS']]
        }
        
        summary_path = self.data_dir.parent / "data_audit_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📄 Reports saved:")
        print(f"   • Detailed: {report_path}")
        print(f"   • Summary: {summary_path}")
        
        return report_path, summary_path

def main():
    """Run the comprehensive data audit"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                    TCGA DATA AUDIT                       ║
    ║                                                          ║
    ║  🚫 ZERO SYNTHETIC DATA POLICY ENFORCEMENT               ║
    ║  🔍 Comprehensive authenticity verification              ║
    ║  🧹 Identifying contaminated datasets                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Run audit
    auditor = TCGAAudit()
    results = auditor.run_comprehensive_audit()
    
    # Final recommendations
    synthetic_count = len([r for r in results if r['authenticity_flag'] == 'SYNTHETIC'])
    if synthetic_count > 0:
        print(f"\n" + "🚨" * 20)
        print(f"CRITICAL FINDING: {synthetic_count} files contain SYNTHETIC data")
        print(f"🚫 This violates the explicit ZERO synthetic data policy")
        print(f"🧹 These files MUST be purged immediately")
        print(f"📋 See data_audit_report.csv for complete list")
        print(f"🚨" * 20)
    
    print(f"\n✅ Audit complete. Next step: Review report and proceed to Phase 2 (Purge)")

if __name__ == "__main__":
    main()
