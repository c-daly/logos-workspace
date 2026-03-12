#!/usr/bin/env bash
# On-pod search launcher. Required: ANTHROPIC_API_KEY, POD_ID, RUNPOD_API_KEY
# Optional: ROUNDS=10 CPR=3 MAX_EXPERIMENTS=150 TARGET_R5=0.82 CONV_PATIENCE=4
set -e

WORKSPACE=/workspace/vl_jepa
H5=/workspace/msrvtt_embeddings.h5
LOG=$WORKSPACE/search.log

ROUNDS=${ROUNDS:-10}
CPR=${CPR:-3}
MAX_EXPERIMENTS=${MAX_EXPERIMENTS:-150}
TARGET_R5=${TARGET_R5:-0.82}
CONV_PATIENCE=${CONV_PATIENCE:-4}

echo "=== VL-JEPA Autonomous Search ==="
python3 -c "
import torch; n=torch.cuda.device_count(); print(f'GPUs: {n}')
[print(f'  GPU {i}: {torch.cuda.get_device_properties(i).name}') for i in range(n)]
"

pip install -q -r $WORKSPACE/requirements.txt

if [ -n "$POD_ID" ] && [ -n "$RUNPOD_API_KEY" ]; then
    STOP_CMD="curl -s -X POST https://api.runpod.io/v2/pod/$POD_ID/stop -H 'Authorization: Bearer $RUNPOD_API_KEY' && echo 'Pod stop requested.'"
else
    STOP_CMD="echo 'No POD_ID/RUNPOD_API_KEY set -- pod will not auto-stop.'"
fi

TRAIN_CMD="cd $WORKSPACE && python search.py \
  --embeddings $H5 \
  --rounds $ROUNDS --configs-per-round $CPR \
  --max-experiments $MAX_EXPERIMENTS \
  --target-metric val_cosine_sim --target-value $TARGET_R5 \
  --convergence-patience $CONV_PATIENCE \
  --resume --output $WORKSPACE/experiment_log.json \
  --llm-provider anthropic 2>&1 | tee $LOG"

tmux new-session -d -s search "bash -c '$TRAIN_CMD; $STOP_CMD'"

echo ""
echo "Training launched in tmux session 'search'."
echo "  tmux attach -t search"
echo "  tail -f $LOG"
sleep 2 && tail -f $LOG
