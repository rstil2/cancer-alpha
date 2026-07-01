#!/usr/bin/env python3
"""
⚠️  PATENT PROTECTED DEMO SOFTWARE ⚠️

Demo Usage Tracking System
==========================

Simple usage tracking for the cancer genomics AI demo.
This tracks demo usage patterns for patent protection purposes.

Patent Information:
- Patent: Provisional Application No. 63/847,316
- Patent Holder: Dr. R. Craig Stillwell
- Contact: craig.stillwell@gmail.com

Author: Dr. R. Craig Stillwell
Date: July 26, 2025
"""

import datetime
import json
import os
from pathlib import Path
import hashlib
import platform
import getpass

class DemoUsageTracker:
    """Simple usage tracking for patent protection"""
    
    def __init__(self):
        self.log_file = Path(__file__).parent / "demo_usage.log"
        self.session_id = self._generate_session_id()
        
    def _generate_session_id(self):
        """Generate a unique session ID"""
        timestamp = str(datetime.datetime.now())
        user = getpass.getuser()
        machine = platform.node()
        session_data = f"{timestamp}-{user}-{machine}"
        return hashlib.md5(session_data.encode()).hexdigest()[:12]
    
    def log_demo_start(self):
        """Log when demo is started"""
        self._log_event("DEMO_START", {
            "session_id": self.session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "user": getpass.getuser(),
            "platform": platform.system(),
            "node": platform.node(),
            "python_version": platform.python_version()
        })
    
    def log_prediction_made(self, model_type, sample_type, confidence):
        """Log when a prediction is made"""
        self._log_event("PREDICTION", {
            "session_id": self.session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "model_type": model_type,
            "sample_type": sample_type,
            "confidence": confidence
        })
    
    def log_demo_end(self):
        """Log when demo session ends"""
        self._log_event("DEMO_END", {
            "session_id": self.session_id,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def _log_event(self, event_type, data):
        """Write event to log file"""
        try:
            log_entry = {
                "event_type": event_type,
                "data": data
            }
            
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            # Fail silently to not interrupt demo
            pass
    
    def get_usage_summary(self):
        """Get basic usage statistics"""
        if not self.log_file.exists():
            return {"total_sessions": 0, "total_predictions": 0}
        
        sessions = set()
        predictions = 0
        
        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry["event_type"] == "DEMO_START":
                            sessions.add(entry["data"]["session_id"])
                        elif entry["event_type"] == "PREDICTION":
                            predictions += 1
                    except:
                        continue
                        
            return {
                "total_sessions": len(sessions),
                "total_predictions": predictions
            }
        except:
            return {"total_sessions": 0, "total_predictions": 0}

# Global tracker instance
demo_tracker = DemoUsageTracker()
