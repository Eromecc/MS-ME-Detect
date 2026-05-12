#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-qwen_download}"
shift || true
MODELS="${*:-medium large}"

PROJECT_DIR="/vepfs-mlp2/queue010/20252203113/MS-ME-Detect"

tmux new -d -s "$SESSION_NAME" "bash -lc '
cd $PROJECT_DIR
source scripts/env.sh
mkdir -p logs
python scripts/download_models.py --models $MODELS --backend modelscope 2>&1 | tee -a logs/download_${SESSION_NAME}.log
'"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Detach with: Ctrl-b then d"
