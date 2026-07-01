#!/usr/bin/env python3
"""
Automated Progress Reporter for Clean TCGA Downloader
====================================================

Monitors the clean TCGA download progress and provides automated status updates
at regular intervals without requiring manual intervention.

Features:
- Real-time sample count monitoring
- Download rate calculation
- ETA estimation
- Process status checking
- Milestone notifications
- Auto-completion detection
"""

import os
import time
import subprocess
import glob
from datetime import datetime, timedelta
import signal
import sys

class AutomatedProgressReporter:
    def __init__(self, base_dir="/Users/stillwell/projects/cancer-alpha"):
        self.base_dir = base_dir
        self.raw_tcga_dir = os.path.join(base_dir, "data/raw_tcga")
        self.target_samples = 50000
        self.check_interval = 300  # 5 minutes
        self.last_count = 0
        self.last_check_time = time.time()
        self.milestones = [35000, 40000, 45000, 49000, 50000]
        self.milestone_hit = set()
        
    def count_downloaded_files(self):
        """Count all downloaded TCGA files"""
        try:
            result = subprocess.run(
                ["find", self.raw_tcga_dir, "-type", "f"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            return 0
        except Exception:
            return 0
    
    def get_directory_size(self):
        """Get total size of downloaded data"""
        try:
            result = subprocess.run(
                ["du", "-sh", self.raw_tcga_dir],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return result.stdout.split()[0]
            return "Unknown"
        except Exception:
            return "Unknown"
    
    def check_downloader_status(self):
        """Check if the downloader process is running"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "clean_tcga_downloader.py"],
                capture_output=True, text=True
            )
            return bool(result.stdout.strip())
        except Exception:
            return False
    
    def calculate_download_rate(self, current_count):
        """Calculate files per minute download rate"""
        current_time = time.time()
        time_diff = (current_time - self.last_check_time) / 60  # minutes
        
        if time_diff > 0 and self.last_count > 0:
            files_added = current_count - self.last_count
            rate = files_added / time_diff
            return rate
        return 0.0
    
    def estimate_completion(self, current_count, rate):
        """Estimate time to completion"""
        if rate > 0:
            remaining_samples = self.target_samples - current_count
            minutes_remaining = remaining_samples / rate
            eta = datetime.now() + timedelta(minutes=minutes_remaining)
            return eta.strftime("%Y-%m-%d %H:%M:%S"), minutes_remaining / 60
        return "Calculating...", 0
    
    def check_milestones(self, current_count):
        """Check and report milestone achievements"""
        for milestone in self.milestones:
            if current_count >= milestone and milestone not in self.milestone_hit:
                self.milestone_hit.add(milestone)
                return milestone
        return None
    
    def generate_progress_report(self):
        """Generate comprehensive progress report"""
        current_count = self.count_downloaded_files()
        current_size = self.get_directory_size()
        is_running = self.check_downloader_status()
        rate = self.calculate_download_rate(current_count)
        eta_time, eta_hours = self.estimate_completion(current_count, rate)
        milestone = self.check_milestones(current_count)
        
        # Update tracking variables
        self.last_count = current_count
        self.last_check_time = time.time()
        
        # Calculate progress percentage
        progress_pct = (current_count / self.target_samples) * 100
        remaining = self.target_samples - current_count
        
        # Generate status report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "="*70)
        print("🔬 TCGA CANCER-ALPHA AUTOMATED PROGRESS REPORT")
        print("="*70)
        print(f"⏰ TIMESTAMP: {timestamp}")
        print(f"🎯 TARGET: {self.target_samples:,} authentic TCGA samples")
        print(f"📊 CURRENT: {current_count:,} samples ({progress_pct:.1f}%)")
        print(f"📈 REMAINING: {remaining:,} samples")
        print(f"💾 DATA SIZE: {current_size}")
        print()
        
        # Progress bar
        bar_length = 50
        filled_length = int(bar_length * progress_pct / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        print(f"PROGRESS: [{bar}] {progress_pct:.1f}%")
        print()
        
        # Status and performance
        status = "🔄 RUNNING" if is_running else "⏸️ STOPPED"
        print(f"🔄 STATUS: {status}")
        print(f"📥 DOWNLOAD RATE: {rate:.1f} files/minute")
        print(f"⏳ ETA: {eta_time}")
        if eta_hours > 0:
            print(f"⏳ TIME REMAINING: ~{eta_hours:.1f} hours")
        print()
        
        # Milestone notifications
        if milestone:
            print("🎉" + "="*65)
            print(f"🏆 MILESTONE ACHIEVED: {milestone:,} SAMPLES!")
            print("🎉" + "="*65)
            print()
        
        # Completion check
        if current_count >= self.target_samples:
            print("🎊" + "="*65)
            print("✅ DOWNLOAD COMPLETE! TARGET REACHED!")
            print(f"🎯 FINAL COUNT: {current_count:,} authentic TCGA samples")
            print(f"💾 TOTAL SIZE: {current_size}")
            print("🎊" + "="*65)
            return True  # Signal completion
            
        print("="*70)
        print("📊 Next update in 5 minutes...")
        print("="*70)
        return False  # Continue monitoring
    
    def run_monitoring(self):
        """Main monitoring loop"""
        print("🚀 Starting Automated TCGA Download Progress Reporter")
        print(f"📊 Monitoring every {self.check_interval//60} minutes")
        print("🎯 Target: 50,000 authentic TCGA samples")
        print("🔄 Press Ctrl+C to stop monitoring\n")
        
        # Initial report
        if self.generate_progress_report():
            print("✅ Download completed! Monitoring stopped.")
            return
        
        try:
            while True:
                time.sleep(self.check_interval)
                
                # Generate progress report
                if self.generate_progress_report():
                    print("✅ Download completed! Monitoring stopped.")
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user.")
            print("🔄 Download continues in background if process is running.")

def signal_handler(sig, frame):
    print('\n⏹️ Automated monitoring stopped.')
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and run the automated reporter
    reporter = AutomatedProgressReporter()
    reporter.run_monitoring()
