#!/bin/bash
# Launch Oncura Streamlit demo from repository root
cd "$(dirname "$0")/demo" || exit 1
exec ./start_demo.sh
