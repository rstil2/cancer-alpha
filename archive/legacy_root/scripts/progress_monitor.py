#!/usr/bin/env python3
"""
Real-time Progress Monitor for TCGA Cancer-Alpha Downloads
Shows live progress toward 50K sample goal with detailed metrics
"""

import os
import time
import subprocess
from datetime import datetime
import glob

TARGET_SAMPLES = 50000
DATA_DIR = "/Users/stillwell/projects/cancer-alpha/data/raw_tcga"
LOG_FILE = "/Users/stillwell/projects/cancer-alpha/downloader.log"

def get_sample_count():
    """Count total .tsv files in all directories"""
    pattern = os.path.join(DATA_DIR, "**", "*.tsv")
    files = glob.glob(pattern, recursive=True)
    return len(files)

def get_recent_activity():
    """Count files created in last 5 minutes"""
    try:
        cmd = f'find {DATA_DIR} -name "*.tsv" -newermt "5 minutes ago" | wc -l'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return int(result.stdout.strip())
    except:
        return 0

def get_current_project():
    """Extract current project from log file"""
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines[-20:]):  # Check last 20 lines
                    if "TCGA-" in line and "✅" in line:
                        # Extract TCGA project code
                        parts = line.split("TCGA-")
                        if len(parts) > 1:
                            project_part = parts[1].split(":")[0]
                            return f"TCGA-{project_part.split()[0]}"
    except:
        pass
    return "Unknown"

def get_process_status():
    """Check if downloader process is running"""
    try:
        result = subprocess.run(
            "ps aux | grep clean_tcga_downloader.py | grep -v grep", 
            shell=True, capture_output=True, text=True
        )
        if result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                parts = lines[0].split()
                cpu = parts[2] if len(parts) > 2 else "0.0"
                return f"Running (CPU: {cpu}%)"
        return "Stopped"
    except:
        return "Unknown"

def get_download_rate():
    """Calculate download rate per minute"""
    recent = get_recent_activity()
    return recent / 5  # Files per minute (5-minute window)

def print_progress_bar(current, target, width=50):
    """Print ASCII progress bar"""
    percentage = (current / target) * 100
    filled = int(width * current / target)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percentage:.1f}%"

def main():
    """Main monitoring loop"""
    print("\n🔬 TCGA Cancer-Alpha Progress Monitor")
    print("="*60)
    print("📊 Real-time progress toward 50,000 authentic samples")
    print("🚫 Zero synthetic policy enforced")
    print("="*60)
    
    try:
        while True:
            # Clear screen (cross-platform)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Header
            print("\n🔬 TCGA CANCER-ALPHA PROGRESS MONITOR")
            print("="*60)
            
            # Get current metrics
            current_samples = get_sample_count()
            remaining = TARGET_SAMPLES - current_samples
            percentage = (current_samples / TARGET_SAMPLES) * 100
            current_project = get_current_project()
            process_status = get_process_status()
            recent_activity = get_recent_activity()
            download_rate = get_download_rate()
            
            # Progress display
            print(f"🎯 TARGET: {TARGET_SAMPLES:,} authentic TCGA samples")
            print(f"📊 CURRENT: {current_samples:,} samples ({percentage:.1f}%)")
            print(f"📈 REMAINING: {remaining:,} samples")
            print()
            
            # Progress bar
            progress_bar = print_progress_bar(current_samples, TARGET_SAMPLES)
            print(f"PROGRESS: {progress_bar}")
            print()
            
            # Current activity
            print(f"🔄 STATUS: {process_status}")
            print(f"🧬 CURRENT PROJECT: {current_project}")
            print(f"⚡ RECENT ACTIVITY: {recent_activity} files (last 5 min)")
            print(f"📥 DOWNLOAD RATE: {download_rate:.1f} files/minute")
            print()
            
            # Time and ETA
            print(f"⏰ LAST UPDATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if download_rate > 0:
                eta_minutes = remaining / download_rate
                eta_hours = eta_minutes / 60
                if eta_hours < 24:
                    print(f"⏳ ETA: ~{eta_hours:.1f} hours (at current rate)")
                else:
                    eta_days = eta_hours / 24
                    print(f"⏳ ETA: ~{eta_days:.1f} days (at current rate)")
            else:
                print("⏳ ETA: Calculating...")
            
            print("\n" + "="*60)
            print("Press Ctrl+C to exit monitor")
            
            # Wait 30 seconds before next update
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitor stopped. Download continues in background.")
        print("🎯 Run 'python progress_monitor.py' to restart monitoring.")

if __name__ == "__main__":
    main()
