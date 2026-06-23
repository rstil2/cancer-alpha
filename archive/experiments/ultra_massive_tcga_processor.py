#!/usr/bin/env python3
"""
Ultra-Massive TCGA Multi-Omics Processor - Scale to 50,000+ Samples
================================================================

Production-scale processor for massive TCGA datasets with:
- Memory-efficient batch processing
- Progress tracking with ETA
- Multi-omics feature engineering
- Robust error handling and recovery
- Distributed processing capabilities
- Real-time monitoring
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import gc
import psutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import gzip
import pickle

class MemoryMonitor:
    """Monitor memory usage and manage resources"""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        return self.process.memory_info().rss / (1024 ** 3)
        
    def check_memory_limit(self) -> bool:
        """Check if we're approaching memory limit"""
        return self.process.memory_info().rss > (self.max_memory_bytes * 0.8)
        
    def force_gc(self):
        """Force garbage collection"""
        gc.collect()

class ProgressTracker:
    """Enhanced progress tracker for massive datasets"""
    
    def __init__(self, logger, save_path: str = None):
        self.logger = logger
        self.save_path = save_path
        self.start_time = time.time()
        self.stages = {}
        self.overall_progress = {'total': 0, 'completed': 0}
        
    def set_overall_total(self, total: int):
        """Set total number of items to process"""
        self.overall_progress['total'] = total
        
    def update_overall(self, completed: int = None):
        """Update overall progress"""
        if completed is not None:
            self.overall_progress['completed'] = completed
        else:
            self.overall_progress['completed'] += 1
            
        progress = self.overall_progress['completed'] / max(self.overall_progress['total'], 1)
        elapsed = time.time() - self.start_time
        
        if progress > 0:
            eta = (elapsed / progress) - elapsed
            self.logger.info(f"🔄 Overall Progress: {self.overall_progress['completed']:,}/{self.overall_progress['total']:,} "
                           f"({progress:.1%}) - ETA: {eta/60:.1f} min")
    
    def start_stage(self, stage_name: str, total_steps: int = 1):
        """Start tracking a new stage"""
        self.stages[stage_name] = {
            'start_time': time.time(),
            'total_steps': total_steps,
            'completed_steps': 0
        }
        self.logger.info(f"🚀 Starting {stage_name} ({total_steps:,} items)...")
        
    def update_stage(self, stage_name: str, completed_steps: int = None):
        """Update progress for a stage"""
        if stage_name not in self.stages:
            return
            
        if completed_steps is not None:
            self.stages[stage_name]['completed_steps'] = completed_steps
        else:
            self.stages[stage_name]['completed_steps'] += 1
            
        stage = self.stages[stage_name]
        elapsed = time.time() - stage['start_time']
        progress = stage['completed_steps'] / stage['total_steps']
        
        if progress > 0 and stage['completed_steps'] % 100 == 0:  # Log every 100 items
            eta = elapsed / progress - elapsed
            rate = stage['completed_steps'] / elapsed if elapsed > 0 else 0
            self.logger.info(f"📊 {stage_name}: {stage['completed_steps']:,}/{stage['total_steps']:,} "
                           f"({progress:.1%}) - Rate: {rate:.1f}/s - ETA: {eta/60:.1f} min")
        
    def complete_stage(self, stage_name: str):
        """Complete a stage"""
        if stage_name not in self.stages:
            return
            
        elapsed = time.time() - self.stages[stage_name]['start_time']
        rate = self.stages[stage_name]['total_steps'] / elapsed if elapsed > 0 else 0
        self.logger.info(f"✅ {stage_name} completed in {elapsed/60:.1f} min (Rate: {rate:.1f}/s)")

class UltraMassiveTCGAProcessor:
    """Ultra-massive TCGA processor for 50,000+ samples"""
    
    def __init__(self, 
                 data_dirs: List[str],
                 output_dir: str = "data/ultra_massive_processed",
                 batch_size: int = 1000,
                 max_memory_gb: float = 8.0,
                 n_workers: int = 4):
        
        self.data_dirs = [Path(d) for d in data_dirs]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.n_workers = n_workers
        
        # Setup monitoring
        self.memory_monitor = MemoryMonitor(max_memory_gb)
        self.setup_logging()
        self.tracker = ProgressTracker(self.logger, str(self.output_dir / "progress.json"))
        
        # Data storage
        self.sample_registry = {}
        self.feature_stats = defaultdict(dict)
        self.batch_counter = 0
        
        self.logger.info("🚀 Ultra-Massive TCGA Processor initialized")
        self.logger.info(f"📂 Data directories: {len(self.data_dirs)}")
        self.logger.info(f"💾 Max memory: {max_memory_gb} GB")
        self.logger.info(f"👥 Workers: {n_workers}")
        self.logger.info(f"📦 Batch size: {batch_size:,}")
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.output_dir / f"ultra_massive_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def discover_files(self) -> Dict[str, List[Path]]:
        """Discover all available files across data directories"""
        self.tracker.start_stage("File Discovery", len(self.data_dirs))
        
        all_files = defaultdict(list)
        
        for data_dir in self.data_dirs:
            self.logger.info(f"🔍 Scanning {data_dir}...")
            
            if not data_dir.exists():
                self.logger.warning(f"⚠️ Directory not found: {data_dir}")
                continue
                
            # Scan for different omics types
            omics_patterns = {
                'mutations': ['mutations/**/*.maf*', 'mutations/**/*.gz'],
                'expression': ['expression/**/*.tsv*', 'expression/**/*.gz'],
                'copy_number': ['copy_number/**/*.tsv*', 'copy_number/**/*.gz'],
                'methylation': ['methylation/**/*.tsv*', 'methylation/**/*.gz'],
                'protein': ['protein/**/*.tsv*', 'protein/**/*.gz'],
                'clinical': ['clinical/**/*.tsv*', 'clinical/**/*.gz', 'clinical/**/*.json']
            }
            
            for omics_type, patterns in omics_patterns.items():
                for pattern in patterns:
                    files = list(data_dir.glob(pattern))
                    all_files[omics_type].extend(files)
                    
            self.tracker.update_stage("File Discovery")
            
        # Log discovery results
        total_files = sum(len(files) for files in all_files.values())
        self.logger.info(f"🎯 File Discovery Complete - Total: {total_files:,} files")
        
        for omics_type, files in all_files.items():
            self.logger.info(f"   {omics_type}: {len(files):,} files")
            
        self.tracker.complete_stage("File Discovery")
        return dict(all_files)
        
    def extract_sample_id(self, filepath: Path) -> Optional[str]:
        """Extract TCGA sample ID from filepath"""
        filename = filepath.name
        
        # Try different TCGA ID patterns
        tcga_patterns = [
            r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]',  # Full TCGA barcode
            r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}',                 # Patient ID
        ]
        
        import re
        for pattern in tcga_patterns:
            match = re.search(pattern, filename)
            if match:
                # Convert to consistent format (first 12 characters)
                tcga_id = match.group(0)[:12]  # TCGA-XX-XXXX format
                return tcga_id
                
        return None
        
    def process_mutation_file(self, filepath: Path) -> Dict[str, Any]:
        """Process a single mutation file"""
        try:
            sample_id = self.extract_sample_id(filepath)
            if not sample_id:
                return {'error': f'No sample ID found in {filepath}'}
                
            # Read mutation file
            if filepath.suffix == '.gz':
                df = pd.read_csv(filepath, sep='\t', compression='gzip', low_memory=False)
            else:
                df = pd.read_csv(filepath, sep='\t', low_memory=False)
                
            if df.empty:
                return {'sample_id': sample_id, 'error': 'Empty file'}
                
            # Extract mutation features
            features = {
                'sample_id': sample_id,
                'mut_total_mutations': len(df),
                'mut_missense': len(df[df['Variant_Classification'] == 'Missense_Mutation']) if 'Variant_Classification' in df.columns else 0,
                'mut_nonsense': len(df[df['Variant_Classification'] == 'Nonsense_Mutation']) if 'Variant_Classification' in df.columns else 0,
                'mut_silent': len(df[df['Variant_Classification'] == 'Silent']) if 'Variant_Classification' in df.columns else 0,
                'mut_frame_shift': len(df[df['Variant_Classification'].isin(['Frame_Shift_Del', 'Frame_Shift_Ins'])]) if 'Variant_Classification' in df.columns else 0,
                'mut_unique_genes': df['Hugo_Symbol'].nunique() if 'Hugo_Symbol' in df.columns else 0,
                'mut_snp': len(df[df['Variant_Type'] == 'SNP']) if 'Variant_Type' in df.columns else 0,
                'mut_dnp': len(df[df['Variant_Type'] == 'DNP']) if 'Variant_Type' in df.columns else 0,
                'mut_ins': len(df[df['Variant_Type'] == 'INS']) if 'Variant_Type' in df.columns else 0,
                'mut_del': len(df[df['Variant_Type'] == 'DEL']) if 'Variant_Type' in df.columns else 0,
            }
            
            return features
            
        except Exception as e:
            return {'error': f'Error processing {filepath}: {str(e)}'}
            
    def process_expression_file(self, filepath: Path) -> Dict[str, Any]:
        """Process a single expression file"""
        try:
            sample_id = self.extract_sample_id(filepath)
            if not sample_id:
                return {'error': f'No sample ID found in {filepath}'}
                
            # Read expression file
            if filepath.suffix == '.gz':
                df = pd.read_csv(filepath, sep='\t', compression='gzip', low_memory=False, index_col=0)
            else:
                df = pd.read_csv(filepath, sep='\t', low_memory=False, index_col=0)
                
            if df.empty:
                return {'sample_id': sample_id, 'error': 'Empty file'}
                
            # Get expression values (assume first column after index)
            expr_values = df.iloc[:, 0].dropna()
            
            if len(expr_values) == 0:
                return {'sample_id': sample_id, 'error': 'No valid expression values'}
                
            # Extract expression features
            features = {
                'sample_id': sample_id,
                'expr_mean': float(expr_values.mean()),
                'expr_std': float(expr_values.std()),
                'expr_median': float(expr_values.median()),
                'expr_q75': float(expr_values.quantile(0.75)),
                'expr_q25': float(expr_values.quantile(0.25)),
                'expr_min': float(expr_values.min()),
                'expr_max': float(expr_values.max()),
                'expr_range': float(expr_values.max() - expr_values.min()),
                'expr_genes_measured': len(expr_values),
                'expr_high_expr': len(expr_values[expr_values > expr_values.quantile(0.9)]),
                'expr_low_expr': len(expr_values[expr_values < expr_values.quantile(0.1)]),
                'expr_zero_expr': len(expr_values[expr_values == 0]),
            }
            
            return features
            
        except Exception as e:
            return {'error': f'Error processing {filepath}: {str(e)}'}
            
    def process_copy_number_file(self, filepath: Path) -> Dict[str, Any]:
        """Process a single copy number file"""
        try:
            sample_id = self.extract_sample_id(filepath)
            if not sample_id:
                return {'error': f'No sample ID found in {filepath}'}
                
            # Read copy number file
            if filepath.suffix == '.gz':
                df = pd.read_csv(filepath, sep='\t', compression='gzip', low_memory=False)
            else:
                df = pd.read_csv(filepath, sep='\t', low_memory=False)
                
            if df.empty:
                return {'sample_id': sample_id, 'error': 'Empty file'}
                
            # Extract copy number features
            # Assume columns: Chromosome, Start, End, Segment_Mean
            if 'Segment_Mean' in df.columns:
                seg_values = df['Segment_Mean'].dropna()
            elif len(df.columns) >= 4:
                seg_values = df.iloc[:, -1].dropna()  # Last column
            else:
                return {'sample_id': sample_id, 'error': 'No segment values found'}
                
            features = {
                'sample_id': sample_id,
                'cn_num_segments': len(seg_values),
                'cn_mean_value': float(seg_values.mean()),
                'cn_median_value': float(seg_values.median()),
                'cn_std_value': float(seg_values.std()),
                'cn_min_value': float(seg_values.min()),
                'cn_max_value': float(seg_values.max()),
                'cn_amplifications': len(seg_values[seg_values > 0.3]),
                'cn_deletions': len(seg_values[seg_values < -0.3]),
                'cn_neutral': len(seg_values[(seg_values >= -0.3) & (seg_values <= 0.3)]),
                'cn_high_amp': len(seg_values[seg_values > 1.0]),
                'cn_deep_del': len(seg_values[seg_values < -1.0]),
            }
            
            # Segment length features if available
            if all(col in df.columns for col in ['Start', 'End']):
                lengths = df['End'] - df['Start']
                features.update({
                    'cn_mean_seg_length': float(lengths.mean()),
                    'cn_median_seg_length': float(lengths.median()),
                    'cn_total_coverage': int(lengths.sum()),
                    'cn_long_segments': len(lengths[lengths > lengths.quantile(0.9)]),
                    'cn_short_segments': len(lengths[lengths < lengths.quantile(0.1)]),
                })
            
            return features
            
        except Exception as e:
            return {'error': f'Error processing {filepath}: {str(e)}'}
            
    def process_protein_file(self, filepath: Path) -> Dict[str, Any]:
        """Process a single protein file"""
        try:
            sample_id = self.extract_sample_id(filepath)
            if not sample_id:
                return {'error': f'No sample ID found in {filepath}'}
                
            # Read protein file
            if filepath.suffix == '.gz':
                df = pd.read_csv(filepath, sep='\t', compression='gzip', low_memory=False, index_col=0)
            else:
                df = pd.read_csv(filepath, sep='\t', low_memory=False, index_col=0)
                
            if df.empty:
                return {'sample_id': sample_id, 'error': 'Empty file'}
                
            # Get protein values (assume first column after index)
            protein_values = df.iloc[:, 0].dropna()
            
            if len(protein_values) == 0:
                return {'sample_id': sample_id, 'error': 'No valid protein values'}
                
            # Extract protein features
            features = {
                'sample_id': sample_id,
                'protein_mean': float(protein_values.mean()),
                'protein_std': float(protein_values.std()),
                'protein_median': float(protein_values.median()),
                'protein_q75': float(protein_values.quantile(0.75)),
                'protein_q25': float(protein_values.quantile(0.25)),
                'protein_min': float(protein_values.min()),
                'protein_max': float(protein_values.max()),
                'protein_range': float(protein_values.max() - protein_values.min()),
                'protein_positive_ratio': float((protein_values > 0).mean()),
                'protein_negative_ratio': float((protein_values < 0).mean()),
                'protein_high_expr': len(protein_values[protein_values > protein_values.quantile(0.9)]),
                'protein_low_expr': len(protein_values[protein_values < protein_values.quantile(0.1)]),
                'protein_extreme_high': len(protein_values[protein_values > protein_values.quantile(0.95)]),
                'protein_extreme_low': len(protein_values[protein_values < protein_values.quantile(0.05)]),
                'protein_total_measured': len(protein_values),
            }
            
            return features
            
        except Exception as e:
            return {'error': f'Error processing {filepath}: {str(e)}'}
            
    def process_files_batch(self, files_batch: List[Tuple[str, Path]]) -> List[Dict[str, Any]]:
        """Process a batch of files"""
        results = []
        
        for omics_type, filepath in files_batch:
            try:
                if omics_type == 'mutations':
                    result = self.process_mutation_file(filepath)
                elif omics_type == 'expression':
                    result = self.process_expression_file(filepath)
                elif omics_type == 'copy_number':
                    result = self.process_copy_number_file(filepath)
                elif omics_type == 'protein':
                    result = self.process_protein_file(filepath)
                else:
                    continue  # Skip unsupported types for now
                    
                results.append({**result, 'omics_type': omics_type})
                
            except Exception as e:
                results.append({'error': f'Batch error processing {filepath}: {str(e)}'})
                
        return results
        
    def save_batch_results(self, batch_results: List[Dict[str, Any]], batch_id: int):
        """Save batch results to disk"""
        batch_file = self.output_dir / f"batch_{batch_id:06d}.pkl"
        
        with open(batch_file, 'wb') as f:
            pickle.dump(batch_results, f)
            
        return batch_file
        
    def process_all_files(self, files_by_type: Dict[str, List[Path]]) -> str:
        """Process all files with batch processing and progress tracking"""
        
        # Prepare file queue
        all_file_tasks = []
        for omics_type, files in files_by_type.items():
            for filepath in files:
                all_file_tasks.append((omics_type, filepath))
                
        total_files = len(all_file_tasks)
        self.logger.info(f"🎯 Processing {total_files:,} files in batches of {self.batch_size:,}")
        
        self.tracker.set_overall_total(total_files)
        self.tracker.start_stage("File Processing", total_files)
        
        # Process in batches
        batch_files = []
        processed_files = 0
        
        for batch_start in range(0, total_files, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_files)
            batch_tasks = all_file_tasks[batch_start:batch_end]
            
            self.logger.info(f"📦 Processing batch {self.batch_counter + 1}: files {batch_start:,} to {batch_end:,}")
            
            # Process batch
            batch_results = self.process_files_batch(batch_tasks)
            
            # Save batch results
            batch_file = self.save_batch_results(batch_results, self.batch_counter)
            batch_files.append(batch_file)
            
            processed_files += len(batch_tasks)
            self.batch_counter += 1
            
            # Update progress
            self.tracker.update_stage("File Processing", processed_files)
            self.tracker.update_overall(processed_files)
            
            # Memory management
            memory_usage = self.memory_monitor.get_memory_usage()
            self.logger.info(f"💾 Memory usage: {memory_usage:.1f} GB")
            
            if self.memory_monitor.check_memory_limit():
                self.logger.warning("⚠️ Approaching memory limit, forcing garbage collection...")
                self.memory_monitor.force_gc()
                
            # Save progress periodically
            if self.batch_counter % 10 == 0:
                progress_data = {
                    'batch_files': [str(bf) for bf in batch_files],
                    'processed_files': processed_files,
                    'total_files': total_files,
                    'timestamp': datetime.now().isoformat()
                }
                
                progress_file = self.output_dir / "processing_progress.json"
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
                    
        self.tracker.complete_stage("File Processing")
        
        # Return batch files list
        batch_list_file = self.output_dir / "batch_files.json"
        with open(batch_list_file, 'w') as f:
            json.dump([str(bf) for bf in batch_files], f, indent=2)
            
        return str(batch_list_file)
        
    def consolidate_batches(self, batch_list_file: str) -> str:
        """Consolidate all batch results into final dataset"""
        self.tracker.start_stage("Batch Consolidation", 1)
        
        with open(batch_list_file, 'r') as f:
            batch_files = json.load(f)
            
        self.logger.info(f"🔄 Consolidating {len(batch_files)} batch files...")
        
        # Consolidate by sample
        sample_data = defaultdict(dict)
        omics_counts = defaultdict(int)
        error_count = 0
        
        for batch_file in batch_files:
            try:
                with open(batch_file, 'rb') as f:
                    batch_results = pickle.load(f)
                    
                for result in batch_results:
                    if 'error' in result:
                        error_count += 1
                        continue
                        
                    sample_id = result.get('sample_id')
                    omics_type = result.get('omics_type')
                    
                    if not sample_id or not omics_type:
                        continue
                        
                    # Store sample data
                    if sample_id not in sample_data:
                        sample_data[sample_id] = {'sample_id': sample_id}
                        
                    # Add omics-specific features
                    for key, value in result.items():
                        if key not in ['sample_id', 'omics_type', 'error']:
                            sample_data[sample_id][key] = value
                            
                    omics_counts[omics_type] += 1
                    
            except Exception as e:
                self.logger.error(f"Error loading batch file {batch_file}: {e}")
                
        self.logger.info(f"✅ Consolidated data:")
        self.logger.info(f"   Total samples: {len(sample_data):,}")
        self.logger.info(f"   Errors: {error_count:,}")
        
        for omics_type, count in omics_counts.items():
            self.logger.info(f"   {omics_type}: {count:,} samples")
            
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(sample_data, orient='index')
        
        # Add cancer type extraction from sample ID
        df['cancer_type'] = df['sample_id'].str[:7]  # TCGA-XX format
        
        # Save final dataset
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"tcga_ultra_massive_{timestamp}.csv"
        
        df.to_csv(output_file, index=False)
        
        # Save metadata
        metadata = {
            'total_samples': len(df),
            'total_features': len(df.columns),
            'omics_coverage': dict(omics_counts),
            'cancer_types': df['cancer_type'].value_counts().to_dict(),
            'processing_time': time.time() - self.tracker.start_time,
            'batch_files_processed': len(batch_files),
            'errors_encountered': error_count,
            'timestamp': timestamp
        }
        
        metadata_file = self.output_dir / f"tcga_ultra_massive_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.tracker.complete_stage("Batch Consolidation")
        
        self.logger.info(f"🎉 Ultra-massive dataset created!")
        self.logger.info(f"📊 Final dataset: {len(df):,} samples, {len(df.columns):,} features")
        self.logger.info(f"💾 Saved to: {output_file}")
        self.logger.info(f"📋 Metadata: {metadata_file}")
        
        return str(output_file)
        
    def run_ultra_massive_processing(self) -> str:
        """Run the complete ultra-massive processing pipeline"""
        self.logger.info("🚀 Starting Ultra-Massive TCGA Processing Pipeline")
        self.logger.info("=" * 80)
        
        try:
            # Step 1: Discover all files
            files_by_type = self.discover_files()
            
            if not any(files_by_type.values()):
                raise ValueError("No files found to process")
                
            # Step 2: Process all files in batches
            batch_list_file = self.process_all_files(files_by_type)
            
            # Step 3: Consolidate results
            final_dataset = self.consolidate_batches(batch_list_file)
            
            # Step 4: Cleanup batch files (optional)
            # self.cleanup_batch_files(batch_list_file)
            
            total_time = time.time() - self.tracker.start_time
            self.logger.info(f"🎉 Ultra-Massive Processing Complete!")
            self.logger.info(f"⏱️ Total processing time: {total_time/3600:.1f} hours")
            
            return final_dataset
            
        except Exception as e:
            self.logger.error(f"❌ Ultra-massive processing failed: {str(e)}")
            raise


def main():
    """Main execution function"""
    # Configuration for 50,000+ sample processing
    data_directories = [
        "data/production_tcga",
        "data/tcga_ultra_massive",
        "data/tcga_real_fixed"
    ]
    
    output_dir = "data/ultra_massive_processed"
    
    print("🚀 Ultra-Massive TCGA Processing - Scale to 50,000+ Samples")
    print("=" * 70)
    print(f"📂 Data directories: {len(data_directories)}")
    print(f"📊 Target: 50,000+ samples")
    print(f"💾 Output: {output_dir}")
    print()
    
    # Initialize processor
    processor = UltraMassiveTCGAProcessor(
        data_dirs=data_directories,
        output_dir=output_dir,
        batch_size=500,  # Smaller batches for memory management
        max_memory_gb=6.0,  # Conservative memory limit
        n_workers=2  # Conservative parallelism
    )
    
    # Run processing
    final_dataset = processor.run_ultra_massive_processing()
    
    print(f"✅ Ultra-massive dataset created: {final_dataset}")
    
    return processor, final_dataset


if __name__ == "__main__":
    processor, dataset = main()
