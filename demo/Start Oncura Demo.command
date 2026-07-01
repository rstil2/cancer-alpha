#!/bin/bash
# macOS: double-click this file in Finder (opens Terminal and runs the demo).
cd "$(dirname "$0")"
chmod +x start_demo.sh 2>/dev/null || true
exec ./start_demo.sh
