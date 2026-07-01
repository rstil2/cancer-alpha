#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo ""
echo " Oncura Demo"
echo " ==========="
echo " Research only — not for clinical use."
echo ""

find_python() {
  command -v python3 >/dev/null 2>&1 && echo python3 && return 0
  command -v python >/dev/null 2>&1 && echo python && return 0
  return 1
}

if ! PY="$(find_python)"; then
  echo " Python 3.8+ is required."
  echo " Install from https://www.python.org/downloads/"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    open "https://www.python.org/downloads/" 2>/dev/null || true
  fi
  read -r -p "Press Enter to close..."
  exit 1
fi

for f in models/multimodal_real_tcga_random_forest.pkl models/scalers.pkl streamlit_app.py; do
  if [[ ! -f "$f" ]]; then
    echo " Missing $f — unzip the full Oncura-Demo package."
    read -r -p "Press Enter to close..."
    exit 1
  fi
done

echo " Installing dependencies (first run may take a minute)..."
"$PY" -m pip install -q -r requirements_streamlit.txt

pkill -f "streamlit run streamlit_app.py" 2>/dev/null || true
sleep 1

echo ""
echo " Starting demo... browser should open at http://localhost:8501"
echo " Press Ctrl+C in this window to stop."
echo ""

exec "$PY" -m streamlit run streamlit_app.py \
  --server.port 8501 \
  --server.address localhost
