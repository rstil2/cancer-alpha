#!/usr/bin/env python3
"""
Cancer Alpha - System Monitoring Utility
========================================

This utility provides comprehensive system monitoring for the Cancer Alpha project,
including API health checks, model validation, system resources, and performance metrics.

Usage:
    python3 utils/system_monitor.py
    
Features:
- API health monitoring
- Model file validation
- System resource checking
- Performance benchmarking
- Automated reporting
"""

import requests
import json
import time
import psutil
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import subprocess
import platform

class CancerAlphaMonitor:
    def __init__(self, api_url: str = "http://localhost:8001"):
        self.api_url = api_url
        self.start_time = datetime.now()
        
        # Expected model files
        self.expected_models = [
            "results/phase2/random_forest_model.pkl",
            "results/phase2/gradient_boosting_model.pkl", 
            "results/phase2/deep_neural_network_model.pkl",
            "results/phase2/ensemble_model.pkl",
            "results/phase2/scaler.pkl"
        ]
        
        # Expected documentation files
        self.expected_docs = [
            "docs/API_REFERENCE_GUIDE.md",
            "docs/UPDATED_PROJECT_ROADMAP_2025.md",
            "MASTER_INSTALLATION_GUIDE.md",
            "README.md"
        ]

    def check_api_health(self) -> Dict[str, Any]:
        """Check API health and connectivity"""
        try:
            # Check root endpoint
            response = requests.get(f"{self.api_url}/", timeout=10)
            if response.status_code == 200:
                root_data = response.json()
                
                # Check health endpoint
                health_response = requests.get(f"{self.api_url}/health", timeout=10)
                health_data = health_response.json() if health_response.status_code == 200 else {}
                
                # Check models endpoint
                models_response = requests.get(f"{self.api_url}/models/info", timeout=10)
                models_data = models_response.json() if models_response.status_code == 200 else {}
                
                return {
                    "status": "healthy",
                    "api_version": root_data.get("version", "unknown"),
                    "models_loaded": health_data.get("models_loaded", False),
                    "model_performance": health_data.get("model_performance", {}),
                    "loaded_models": models_data.get("loaded_models", []),
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "endpoints_accessible": True
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"API returned status {response.status_code}",
                    "endpoints_accessible": False
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "status": "disconnected",
                "error": "Cannot connect to API",
                "endpoints_accessible": False
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "endpoints_accessible": False
            }

    def check_model_files(self) -> Dict[str, Any]:
        """Validate model files exist and are accessible"""
        model_status = {}
        missing_models = []
        total_size = 0
        
        for model_path in self.expected_models:
            path = Path(model_path)
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                total_size += size_mb
                model_status[model_path] = {
                    "exists": True,
                    "size_mb": round(size_mb, 2),
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                }
            else:
                missing_models.append(model_path)
                model_status[model_path] = {"exists": False}
        
        return {
            "total_models": len(self.expected_models),
            "available_models": len(self.expected_models) - len(missing_models),
            "missing_models": missing_models,
            "total_size_mb": round(total_size, 2),
            "models_detail": model_status,
            "all_models_available": len(missing_models) == 0
        }

    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk usage for current directory
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # System uptime
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time
            
            return {
                "cpu": {
                    "usage_percent": round(cpu_percent, 1),
                    "cores": cpu_count,
                    "status": "good" if cpu_percent < 80 else "high" if cpu_percent < 95 else "critical"
                },
                "memory": {
                    "usage_percent": round(memory_percent, 1),
                    "available_gb": round(memory_available_gb, 2),
                    "total_gb": round(memory_total_gb, 2),
                    "status": "good" if memory_percent < 80 else "high" if memory_percent < 95 else "critical"
                },
                "disk": {
                    "usage_percent": round(disk_percent, 1),
                    "free_gb": round(disk_free_gb, 2),
                    "total_gb": round(disk_total_gb, 2),
                    "status": "good" if disk_percent < 80 else "high" if disk_percent < 95 else "critical"
                },
                "system": {
                    "platform": platform.system(),
                    "python_version": platform.python_version(),
                    "uptime_hours": round(uptime.total_seconds() / 3600, 1)
                }
            }
        except Exception as e:
            return {"error": f"Failed to get system resources: {str(e)}"}

    def test_api_performance(self) -> Dict[str, Any]:
        """Test API performance with sample requests"""
        if not self.check_api_health()["endpoints_accessible"]:
            return {"error": "API not accessible for performance testing"}
        
        try:
            performance_results = {}
            
            # Test health endpoint
            start_time = time.time()
            health_response = requests.get(f"{self.api_url}/health")
            health_time = (time.time() - start_time) * 1000
            
            performance_results["health_endpoint"] = {
                "response_time_ms": round(health_time, 2),
                "status_code": health_response.status_code
            }
            
            # Test models info endpoint
            start_time = time.time()
            models_response = requests.get(f"{self.api_url}/models/info")
            models_time = (time.time() - start_time) * 1000
            
            performance_results["models_endpoint"] = {
                "response_time_ms": round(models_time, 2),
                "status_code": models_response.status_code
            }
            
            # Test cancer types endpoint
            start_time = time.time()
            types_response = requests.get(f"{self.api_url}/cancer-types")
            types_time = (time.time() - start_time) * 1000
            
            performance_results["cancer_types_endpoint"] = {
                "response_time_ms": round(types_time, 2),
                "status_code": types_response.status_code
            }
            
            # Test sample prediction if models are loaded
            if health_response.status_code == 200:
                health_data = health_response.json()
                if health_data.get("models_loaded", False):
                    start_time = time.time()
                    test_response = requests.get(f"{self.api_url}/test-real")
                    test_time = (time.time() - start_time) * 1000
                    
                    performance_results["test_prediction"] = {
                        "response_time_ms": round(test_time, 2),
                        "status_code": test_response.status_code
                    }
            
            # Calculate average response time
            response_times = [r["response_time_ms"] for r in performance_results.values() 
                            if "response_time_ms" in r]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            performance_results["summary"] = {
                "average_response_time_ms": round(avg_response_time, 2),
                "total_tests": len(performance_results) - 1,  # Exclude summary
                "all_successful": all(r.get("status_code") == 200 for r in performance_results.values() 
                                     if "status_code" in r)
            }
            
            return performance_results
            
        except Exception as e:
            return {"error": f"Performance testing failed: {str(e)}"}

    def check_documentation(self) -> Dict[str, Any]:
        """Check if documentation files exist and are up-to-date"""
        doc_status = {}
        missing_docs = []
        
        for doc_path in self.expected_docs:
            path = Path(doc_path)
            if path.exists():
                size_kb = path.stat().st_size / 1024
                modified = datetime.fromtimestamp(path.stat().st_mtime)
                days_old = (datetime.now() - modified).days
                
                doc_status[doc_path] = {
                    "exists": True,
                    "size_kb": round(size_kb, 2),
                    "last_modified": modified.isoformat(),
                    "days_old": days_old,
                    "status": "recent" if days_old < 7 else "outdated" if days_old < 30 else "old"
                }
            else:
                missing_docs.append(doc_path)
                doc_status[doc_path] = {"exists": False}
        
        return {
            "total_docs": len(self.expected_docs),
            "available_docs": len(self.expected_docs) - len(missing_docs),
            "missing_docs": missing_docs,
            "docs_detail": doc_status,
            "all_docs_available": len(missing_docs) == 0
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report"""
        print("🔍 Cancer Alpha System Monitor")
        print("=" * 50)
        print(f"📅 Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Collect all monitoring data
        api_health = self.check_api_health()
        model_files = self.check_model_files()
        system_resources = self.check_system_resources()
        api_performance = self.test_api_performance()
        documentation = self.check_documentation()
        
        # Overall system status
        overall_status = "healthy"
        if not api_health.get("endpoints_accessible", False):
            overall_status = "api_down"
        elif not model_files.get("all_models_available", False):
            overall_status = "models_missing"
        elif not api_health.get("models_loaded", False):
            overall_status = "models_not_loaded"
        elif any(r.get("status") == "critical" for r in system_resources.values() if isinstance(r, dict)):
            overall_status = "resource_critical"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "monitor_version": "1.0.0",
            "overall_status": overall_status,
            "api_health": api_health,
            "model_files": model_files,
            "system_resources": system_resources,
            "api_performance": api_performance,
            "documentation": documentation
        }
        
        return report

    def print_summary(self, report: Dict[str, Any]):
        """Print a human-readable summary of the monitoring report"""
        
        # Overall status
        status_emoji = {
            "healthy": "✅",
            "api_down": "🔴", 
            "models_missing": "⚠️",
            "models_not_loaded": "🟡",
            "resource_critical": "🟠"
        }
        
        print(f"🎯 Overall Status: {status_emoji.get(report['overall_status'], '❓')} {report['overall_status'].replace('_', ' ').title()}")
        print()
        
        # API Status
        api_health = report["api_health"]
        print("🌐 API Status:")
        if api_health.get("endpoints_accessible"):
            print(f"   ✅ API accessible at {self.api_url}")
            print(f"   📊 Version: {api_health.get('api_version', 'unknown')}")
            print(f"   ⚡ Response time: {api_health.get('response_time_ms', 0):.1f}ms")
            print(f"   🤖 Models loaded: {'✅' if api_health.get('models_loaded') else '❌'}")
        else:
            print(f"   ❌ API not accessible: {api_health.get('error', 'unknown error')}")
        print()
        
        # Model Files
        model_files = report["model_files"]
        print("🗂 Model Files:")
        print(f"   📁 Available: {model_files['available_models']}/{model_files['total_models']}")
        print(f"   💾 Total size: {model_files['total_size_mb']:.1f} MB")
        if model_files["missing_models"]:
            print(f"   ⚠️ Missing: {', '.join(model_files['missing_models'])}")
        else:
            print("   ✅ All model files present")
        print()
        
        # System Resources
        system_resources = report["system_resources"]
        if "error" not in system_resources:
            print("💻 System Resources:")
            cpu = system_resources["cpu"]
            memory = system_resources["memory"]
            disk = system_resources["disk"]
            
            print(f"   🖥 CPU: {cpu['usage_percent']}% ({cpu['cores']} cores) - {cpu['status']}")
            print(f"   🧠 Memory: {memory['usage_percent']}% ({memory['available_gb']:.1f}GB free) - {memory['status']}")
            print(f"   💿 Disk: {disk['usage_percent']}% ({disk['free_gb']:.1f}GB free) - {disk['status']}")
            print(f"   ⏱ System uptime: {system_resources['system']['uptime_hours']:.1f} hours")
        else:
            print(f"💻 System Resources: ❌ {system_resources['error']}")
        print()
        
        # Performance
        api_performance = report["api_performance"]
        if "error" not in api_performance and "summary" in api_performance:
            summary = api_performance["summary"]
            print("⚡ API Performance:")
            print(f"   📈 Average response time: {summary['average_response_time_ms']:.1f}ms")
            print(f"   🧪 Tests completed: {summary['total_tests']}")
            print(f"   ✅ All tests successful: {'Yes' if summary['all_successful'] else 'No'}")
        else:
            print(f"⚡ API Performance: ❌ {api_performance.get('error', 'Testing failed')}")
        print()
        
        # Documentation
        documentation = report["documentation"]
        print("📚 Documentation:")
        print(f"   📄 Available: {documentation['available_docs']}/{documentation['total_docs']}")
        if documentation["missing_docs"]:
            print(f"   ⚠️ Missing: {', '.join(documentation['missing_docs'])}")
        else:
            print("   ✅ All documentation files present")
        print()
        
        print("=" * 50)
        print("💡 For detailed information, check the full report JSON")
        print(f"🚀 Cancer Alpha v{api_health.get('api_version', 'unknown')} monitoring complete!")

def main():
    """Main monitoring function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cancer Alpha System Monitor")
    parser.add_argument("--api-url", default="http://localhost:8001", 
                       help="API URL to monitor (default: http://localhost:8001)")
    parser.add_argument("--json", action="store_true", 
                       help="Output full report as JSON")
    parser.add_argument("--continuous", type=int, metavar="SECONDS",
                       help="Run continuously with specified interval in seconds")
    
    args = parser.parse_args()
    
    monitor = CancerAlphaMonitor(api_url=args.api_url)
    
    def run_monitoring():
        report = monitor.generate_report()
        
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            monitor.print_summary(report)
        
        return report
    
    if args.continuous:
        print(f"🔄 Running continuous monitoring every {args.continuous} seconds...")
        print("Press Ctrl+C to stop")
        print()
        
        try:
            while True:
                run_monitoring()
                if not args.json:
                    print(f"\n⏳ Next check in {args.continuous} seconds...\n")
                time.sleep(args.continuous)
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
    else:
        run_monitoring()

if __name__ == "__main__":
    main()
