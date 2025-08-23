#!/usr/bin/env python3
import pickle
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

PROGRESS_PATH = Path("data/production_tcga/download_progress.pkl")

# Stub class so pickle can resolve objects saved from __main__.DownloadProgress
# in production_tcga_downloader.py
@dataclass
class DownloadProgress:
    project: str
    data_type: str
    total_files: int
    downloaded_files: int
    failed_files: int
    start_time: datetime
    last_update: datetime

def summarize(progress):
    total_files = 0
    total_downloaded = 0
    total_failed = 0
    per_project = {}

    for key, p in progress.items():
        proj = p.project
        per_project.setdefault(proj, {"files": 0, "downloaded": 0, "failed": 0})
        per_project[proj]["files"] += p.total_files
        per_project[proj]["downloaded"] += p.downloaded_files
        per_project[proj]["failed"] += p.failed_files
        total_files += p.total_files
        total_downloaded += p.downloaded_files
        total_failed += p.failed_files

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"=== Production TCGA Downloader Progress @ {ts} ===")
    lines.append(f"Overall: total_files={total_files} downloaded={total_downloaded} failed={total_failed} success_rate={(total_downloaded/total_files*100 if total_files else 0):.1f}%")
    for proj in sorted(per_project.keys()):
        stats = per_project[proj]
        rate = (stats["downloaded"]/stats["files"]*100) if stats["files"] else 0
        lines.append(f"  {proj}: total={stats['files']} downloaded={stats['downloaded']} failed={stats['failed']} ({rate:.1f}% done)")
    return "\n".join(lines)


def main():
    if not PROGRESS_PATH.exists():
        print(f"No progress file found at {PROGRESS_PATH}")
        sys.exit(0)
    try:
        with open(PROGRESS_PATH, 'rb') as f:
            progress = pickle.load(f)
    except Exception as e:
        print(f"Failed to read progress: {e}")
        sys.exit(1)

    print(summarize(progress))

if __name__ == "__main__":
    main()

