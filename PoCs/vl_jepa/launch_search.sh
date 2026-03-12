#!/usr/bin/env bash
# Local launch: start pod, upload, run podrun_search.sh.
# Required: POD_ID, RUNPOD_API_KEY, ANTHROPIC_API_KEY
# Optional: SSH_KEY, ROUNDS, CPR, MAX_EXPERIMENTS, TARGET_R5, CONV_PATIENCE
set -e

: "${POD_ID:?Set POD_ID}"
: "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY}"
: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_runpod_key}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# H5 file stays in the old location (not moved to vl_jepa)
H5="$HOME/projects/LOGOS/PoCs/jepa_clip_translator/msrvtt_embeddings.h5"
SSH_USER="${POD_ID}-root"
SSH_HOST="ssh.runpod.io"
SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no"

echo "=== VL-JEPA Launch === Pod: $POD_ID"

echo "--- Starting pod ---"
curl -s -X POST "https://api.runpod.io/v2/pod/$POD_ID/start" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print('Response:', d.get('status', d))"

echo "--- Waiting for SSH (max 5 min) ---"
MAX_WAIT=300; WAITED=0
while true; do
    PORTS=$( curl -s "https://api.runpod.io/v2/pod/$POD_ID" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" | python3 -c \
      "import sys,json; d=json.load(sys.stdin)
pods=d.get('data',{}).get('myself',{}).get('pods',[{}])
p=pods[0] if pods else {}
print(p.get('desiredStatus','?'), str(p.get('runtime',{}).get('ports','')))" 2>/dev/null || echo "?")
    echo "  $PORTS (${WAITED}s elapsed)"
    if echo "$PORTS" | grep -q RUNNING && echo "$PORTS" | grep -q "22"; then break; fi
    sleep 10; WAITED=$((WAITED+10))
    [ $WAITED -ge $MAX_WAIT ] && { echo "Timeout waiting for pod."; exit 1; }
done
sleep 5

echo "--- Uploading vl_jepa/ ---"
$SSH "$SSH_USER@$SSH_HOST" "mkdir -p /workspace/vl_jepa"
rsync -avz --progress -e "$SSH" "$SCRIPT_DIR/" "$SSH_USER@$SSH_HOST:/workspace/vl_jepa/"

echo "--- Uploading msrvtt_embeddings.h5 (skips if unchanged) ---"
rsync -avz --progress --ignore-existing -e "$SSH" "$H5" "$SSH_USER@$SSH_HOST:/workspace/"

echo "--- Launching training ---"
$SSH "$SSH_USER@$SSH_HOST" \
  "POD_ID=$POD_ID RUNPOD_API_KEY=$RUNPOD_API_KEY ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
   ROUNDS=${ROUNDS:-10} CPR=${CPR:-3} MAX_EXPERIMENTS=${MAX_EXPERIMENTS:-150} \
   TARGET_R5=${TARGET_R5:-0.82} CONV_PATIENCE=${CONV_PATIENCE:-4} \
   bash /workspace/vl_jepa/podrun_search.sh &"

echo ""
echo "============================================"
echo "Training launched."
echo "  $SSH $SSH_USER@$SSH_HOST 'tail -f /workspace/vl_jepa/search.log'"
echo "  $SSH $SSH_USER@$SSH_HOST 'tmux attach -t search'"
echo "Artifacts when done:"
echo "  rsync -avz -e '$SSH' $SSH_USER@$SSH_HOST:/workspace/vl_jepa/ ./results/"
echo "============================================"
