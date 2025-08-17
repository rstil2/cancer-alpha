#!/usr/bin/env python3
"""
Simple HTTP server to serve the Oncura demo package locally.
Run this script and then access http://localhost:8000/cancer_genomics_ai_demo.zip
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

PORT = 8000
DEMO_FILE = "cancer_genomics_ai_demo.zip"

def main():
    """Start a simple HTTP server to serve the demo package."""
    
    # Check if demo package exists
    if not os.path.exists(DEMO_FILE):
        print(f"❌ Error: {DEMO_FILE} not found!")
        print("Please run this script from the cancer-alpha directory where the demo package is located.")
        sys.exit(1)
    
    # Get file size for display
    file_size = os.path.getsize(DEMO_FILE)
    file_size_mb = file_size / (1024 * 1024)
    
    print("🌟 Oncura Demo Server")
    print("=" * 40)
    print(f"📦 Demo package: {DEMO_FILE}")
    print(f"📊 File size: {file_size_mb:.1f}MB")
    print(f"🌐 Starting server on port {PORT}...")
    print()
    print(f"🔗 Download URL: http://localhost:{PORT}/{DEMO_FILE}")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 40)
    
    # Start server
    try:
        with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
            print(f"✅ Server running at http://localhost:{PORT}/")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Error: Port {PORT} is already in use.")
            print(f"Try a different port or stop the service using port {PORT}")
        else:
            print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()
