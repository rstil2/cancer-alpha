#!/usr/bin/env python3
"""
🧬 Cancer Alpha API Launcher
Cross-platform Python script to start the Cancer Alpha API cleanly
Handles port conflicts and dependency checking on Windows, macOS, and Linux
"""

import os
import sys
import subprocess
import argparse
import time
import platform
from pathlib import Path

# Configuration
DEFAULT_PORT = 8001
REAL_API_SCRIPT = "real_cancer_alpha_api.py"
DEMO_API_SCRIPT = "simple_cancer_api.py"

class Colors:
    """ANSI color codes for cross-platform colored output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color
    
    @classmethod
    def disable_on_windows(cls):
        """Disable colors on Windows if not supported"""
        if platform.system() == 'Windows' and not os.getenv('ANSICON'):
            cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = cls.NC = ''

# Initialize colors
Colors.disable_on_windows()

def print_status(message):
    print(f"{Colors.BLUE}🧬{Colors.NC} {message}")

def print_success(message):
    print(f"{Colors.GREEN}✅{Colors.NC} {message}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠️{Colors.NC} {message}")

def print_error(message):
    print(f"{Colors.RED}❌{Colors.NC} {message}")

def find_port_process(port):
    """Find process using the specified port (cross-platform)"""
    try:
        if platform.system() == 'Windows':
            result = subprocess.run(
                ['netstat', '-ano'], 
                capture_output=True, text=True, check=True
            )
            for line in result.stdout.split('\n'):
                if f':{port} ' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        return parts[-1]  # PID is last column
        else:
            # macOS and Linux
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'], 
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')[0]
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None

def kill_process(pid):
    """Kill process by PID (cross-platform)"""
    if not pid:
        return True
    
    try:
        if platform.system() == 'Windows':
            subprocess.run(['taskkill', '/PID', pid, '/F'], 
                         capture_output=True, check=True)
        else:
            subprocess.run(['kill', '-9', pid], 
                         capture_output=True, check=True)
        return True
    except subprocess.SubprocessError:
        return False

def cleanup_port(port):
    """Clean up any process using the specified port"""
    print_status(f"Checking port {port}...")
    
    pid = find_port_process(port)
    if pid:
        print_warning(f"Found process {pid} using port {port}")
        print_status(f"Terminating process {pid}...")
        
        if kill_process(pid):
            time.sleep(2)
            # Verify it's really gone
            if find_port_process(port):
                print_error(f"Failed to terminate process on port {port}")
                return False
            else:
                print_success(f"Port {port} is now available")
        else:
            print_error(f"Failed to kill process {pid}")
            return False
    else:
        print_success(f"Port {port} is already available")
    
    return True

def check_dependencies(api_script):
    """Check if all dependencies are available"""
    print_status("Checking dependencies...")
    
    # Check Python
    if sys.version_info < (3, 8):
        print_error("Python 3.8+ is required")
        return False
    
    # Check if API script exists
    if not Path(api_script).exists():
        print_error(f"API script {api_script} not found in current directory")
        print_status(f"Current directory: {os.getcwd()}")
        python_files = list(Path('.').glob('*.py'))
        if python_files:
            print_status("Available Python files:")
            for file in python_files:
                print(f"  {file}")
        else:
            print_status("No Python files found")
        return False
    
    # Check Python packages
    required_packages = ['fastapi', 'uvicorn', 'numpy', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print_error("Missing required Python packages:")
        for package in missing_packages:
            print(f"  {package}")
        print_status("Install with: pip3 install " + " ".join(missing_packages))
        return False
    
    print_success("All dependencies check passed")
    return True

def start_api(api_script, port):
    """Start the Cancer Alpha API"""
    print_status(f"Starting Cancer Alpha API on port {port}...")
    print("=" * 50)
    
    # Set environment variable for custom port
    if port != DEFAULT_PORT:
        os.environ['CANCER_ALPHA_PORT'] = str(port)
    
    try:
        # Start the API using Python subprocess
        subprocess.run([sys.executable, api_script], check=True)
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to start API: {e}")
        return False
    except KeyboardInterrupt:
        print_status("API stopped by user")
        return True
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='🧬 Cancer Alpha API Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/start_api.py                    # Start real models API on port 8001
  python3 scripts/start_api.py --port 8002        # Start on custom port
  python3 scripts/start_api.py --demo             # Start demo API with mock predictions
        """
    )
    
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                       help=f'Port to use (default: {DEFAULT_PORT})')
    parser.add_argument('--demo', action='store_true',
                       help='Start demo API instead of real models API')
    
    args = parser.parse_args()
    
    print("🧬 Cancer Alpha API Launcher")
    print("=" * 40)
    
    # Determine which API script to use
    if args.demo:
        api_script = DEMO_API_SCRIPT
        print_status("Starting DEMO API with mock predictions")
    else:
        api_script = REAL_API_SCRIPT
        print_status("Starting REAL API with trained models")
    
    # Check dependencies
    if not check_dependencies(api_script):
        sys.exit(1)
    
    # Cleanup port
    if not cleanup_port(args.port):
        sys.exit(1)
    
    # Start the API
    if not start_api(api_script, args.port):
        sys.exit(1)

if __name__ == '__main__':
    main()
