#!/bin/bash
set -euo pipefail

echo "Oncura demo — starting Streamlit on http://localhost:8501"
echo "Research only. Not for clinical use."
echo ""

if ! command -v python3 &>/dev/null; then
  echo "Python 3.8+ required."
  exit 1
fi

for f in models/multimodal_real_tcga_random_forest.pkl models/scalers.pkl streamlit_app.py; do
  if [ ! -f "$f" ]; then
    echo "Missing $f — run from the demo/ directory with models present."
    exit 1
  fi
done

python3 -m pip install -q -r requirements_streamlit.txt

pkill -f "streamlit run streamlit_app.py" 2>/dev/null || true
sleep 1

exec python3 -m streamlit run streamlit_app.py \
  --server.port 8501 \
  --server.address localhost
