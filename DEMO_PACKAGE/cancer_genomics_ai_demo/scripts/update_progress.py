#!/usr/bin/env python3

"""
Cancer Alpha Progress Tracking Utility
Updates checkpoint files with task progress
"""

import json
import argparse
import os
from datetime import datetime
from pathlib import Path

class ProgressTracker:
    def __init__(self, checkpoint_file):
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
    def load_progress(self):
        """Load existing progress or create new structure"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        else:
            return self.create_new_progress()
    
    def create_new_progress(self):
        """Create new progress structure"""
        return {
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "phase_a": {
                "status": "not_started",
                "completed_tasks": [],
                "failed_tasks": [],
                "current_task": None,
                "progress_percent": 0,
                "tasks": {
                    "A.1.1": {"status": "pending", "description": "TCGA Data Access Setup"},
                    "A.1.2": {"status": "pending", "description": "Real Data Download Pipeline"},
                    "A.1.3": {"status": "pending", "description": "Data Preprocessing Pipeline"},
                    "A.2.1": {"status": "pending", "description": "Real Data Model Training"},
                    "A.2.2": {"status": "pending", "description": "Clinical Performance Validation"},
                    "A.2.3": {"status": "pending", "description": "Biological Insight Analysis"}
                }
            },
            "phase_b": {
                "status": "not_started",
                "completed_tasks": [],
                "failed_tasks": [],
                "current_task": None,
                "progress_percent": 0,
                "tasks": {
                    "B.1.1": {"status": "pending", "description": "Kubernetes Production Setup"},
                    "B.1.2": {"status": "pending", "description": "Load Balancing and Auto-scaling"},
                    "B.1.3": {"status": "pending", "description": "Database and Storage Scaling"},
                    "B.2.1": {"status": "pending", "description": "HIPAA Compliance Implementation"},
                    "B.2.2": {"status": "pending", "description": "Authentication and Authorization"},
                    "B.2.3": {"status": "pending", "description": "Data Privacy and Anonymization"},
                    "B.3.1": {"status": "pending", "description": "Clinical Dashboard"},
                    "B.3.2": {"status": "pending", "description": "API Gateway and Documentation"},
                    "B.3.3": {"status": "pending", "description": "Integration Testing"}
                }
            },
            "total_progress": 0,
            "estimated_completion": None
        }
    
    def save_progress(self, progress_data):
        """Save progress to checkpoint file"""
        progress_data["last_updated"] = datetime.now().isoformat()
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def update_task(self, phase, task_id, status, progress_percent=None):
        """Update specific task status"""
        progress_data = self.load_progress()
        
        phase_key = f"phase_{phase.lower()}"
        if phase_key not in progress_data:
            print(f"Error: Phase {phase} not found")
            return
        
        phase_data = progress_data[phase_key]
        
        # Update task status
        if task_id in phase_data["tasks"]:
            phase_data["tasks"][task_id]["status"] = status
            phase_data["tasks"][task_id]["last_updated"] = datetime.now().isoformat()
            
            if progress_percent is not None:
                phase_data["tasks"][task_id]["progress"] = progress_percent
        
        # Update phase status and task lists
        if status == "running":
            phase_data["current_task"] = task_id
            if phase_data["status"] == "not_started":
                phase_data["status"] = "in_progress"
        
        elif status == "completed":
            if task_id not in phase_data["completed_tasks"]:
                phase_data["completed_tasks"].append(task_id)
            if task_id in phase_data["failed_tasks"]:
                phase_data["failed_tasks"].remove(task_id)
            phase_data["current_task"] = None
        
        elif status == "failed":
            if task_id not in phase_data["failed_tasks"]:
                phase_data["failed_tasks"].append(task_id)
            if task_id in phase_data["completed_tasks"]:
                phase_data["completed_tasks"].remove(task_id)
            phase_data["current_task"] = None
            phase_data["status"] = "failed"
        
        # Calculate phase progress
        total_tasks = len(phase_data["tasks"])
        completed_tasks = len(phase_data["completed_tasks"])
        phase_data["progress_percent"] = (completed_tasks / total_tasks) * 100
        
        # Check if phase is complete
        if completed_tasks == total_tasks:
            phase_data["status"] = "completed"
            phase_data["completion_time"] = datetime.now().isoformat()
        
        # Calculate total progress
        total_tasks_all = len(progress_data["phase_a"]["tasks"]) + len(progress_data["phase_b"]["tasks"])
        total_completed = len(progress_data["phase_a"]["completed_tasks"]) + len(progress_data["phase_b"]["completed_tasks"])
        progress_data["total_progress"] = (total_completed / total_tasks_all) * 100
        
        self.save_progress(progress_data)
        
        print(f"Updated {phase} task {task_id}: {status}")
        print(f"Phase {phase} progress: {phase_data['progress_percent']:.1f}%")
        print(f"Total progress: {progress_data['total_progress']:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Update Cancer Alpha progress tracking")
    parser.add_argument("--checkpoint-file", required=True, help="Path to checkpoint file")
    parser.add_argument("--phase", required=True, choices=["A", "B"], help="Phase (A or B)")
    parser.add_argument("--task", required=True, help="Task ID (e.g., A.1.1)")
    parser.add_argument("--status", required=True, choices=["pending", "running", "completed", "failed"], help="Task status")
    parser.add_argument("--progress", type=int, help="Progress percentage (0-100)")
    
    args = parser.parse_args()
    
    tracker = ProgressTracker(args.checkpoint_file)
    tracker.update_task(args.phase, args.task, args.status, args.progress)

if __name__ == "__main__":
    main()
