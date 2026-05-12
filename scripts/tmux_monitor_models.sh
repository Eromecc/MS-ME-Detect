#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-qwen_monitor}"

tmux new -d -s "$SESSION_NAME" "bash -lc '
cd /vepfs-mlp2/queue010/20252203113/MS-ME-Detect
source scripts/env.sh
watch -n 60 \"echo ==== processes ====; ps -ef | grep -E '\''download_models|modelscope|hf download|Qwen2.5'\'' | grep -v grep || true; echo; echo ==== sizes ====; du -sh /vepfs-mlp2/queue010/20252203113/models/Qwen2.5-7B 2>/dev/null || true; du -sh /vepfs-mlp2/queue010/20252203113/models/Qwen2.5-14B 2>/dev/null || true; du -sh /vepfs-mlp2/queue010/20252203113/models/Qwen2.5-32B 2>/dev/null || true; echo; echo ==== readiness ====; python scripts/check_models.py --models small medium large || true\"
'"

echo "Started monitor session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Detach with: Ctrl-b then d"
