#!/usr/bin/env bash
# monitor_pod_process.sh — check a process on a remote pod, restart if dead, tail log
#
# Usage:
#   monitor_pod_process.sh [options]
#
# Options:
#   -h HOST         SSH host (required)
#   -p PORT         SSH port (default: 22)
#   -k KEY          SSH key file (default: ~/.ssh/id_rsa)
#   -u USER         SSH user (default: root)
#   -n PATTERN      pgrep pattern to match process (required)
#   -r RESTART_CMD  command to run if process is not found (required)
#   -l LOG_FILE     log file to tail on the remote host (optional)
#   -t TAIL_LINES   number of log lines to tail (default: 10)
#   -s STALE_MINS   minutes since log modification before flagging as hung (default: 30)
#
# Example:
#   monitor_pod_process.sh \
#     -h 216.249.100.66 -p 13669 -k ~/.ssh/id_runpod_key \
#     -n search.py \
#     -r 'source /root/.pod_env; nohup python /workspace/vl_jepa/search.py ... >> search.log 2>&1 &' \
#     -l /workspace/vl_jepa/search.log

set -euo pipefail

HOST=""
PORT=22
KEY=~/.ssh/id_rsa
USER=root
PATTERN=""
RESTART_CMD=""
LOG_FILE=""
TAIL_LINES=10
STALE_MINS=30

while getopts "h:p:k:u:n:r:l:t:s:" opt; do
  case $opt in
    h) HOST="$OPTARG" ;;
    p) PORT="$OPTARG" ;;
    k) KEY="$OPTARG" ;;
    u) USER="$OPTARG" ;;
    n) PATTERN="$OPTARG" ;;
    r) RESTART_CMD="$OPTARG" ;;
    l) LOG_FILE="$OPTARG" ;;
    t) TAIL_LINES="$OPTARG" ;;
    s) STALE_MINS="$OPTARG" ;;
    *) echo "Unknown option: $opt" >&2; exit 1 ;;
  esac
done

if [[ -z "$HOST" || -z "$PATTERN" || -z "$RESTART_CMD" ]]; then
  echo "Error: -h HOST, -n PATTERN, and -r RESTART_CMD are required." >&2
  exit 1
fi

SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 $USER@$HOST"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Checking $PATTERN on $HOST:$PORT"

# Check process
PID=$($SSH "pgrep -f '$PATTERN' | head -1" 2>/dev/null || true)

if [[ -z "$PID" ]]; then
  echo "  DEAD — restarting..."
  $SSH "$RESTART_CMD" || true
  sleep 3
  PID=$($SSH "pgrep -f '$PATTERN' | head -1" 2>/dev/null || true)
  if [[ -n "$PID" ]]; then
    echo "  Restarted (PID $PID)"
  else
    echo "  WARNING: restart attempted but process not found"
  fi
else
  echo "  RUNNING (PID $PID)"
fi

# Check log if provided
if [[ -n "$LOG_FILE" ]]; then
  STALE=$($SSH "
    MOD=\$(stat -c %Y '$LOG_FILE' 2>/dev/null || echo 0)
    NOW=\$(date +%s)
    echo \$(( (NOW - MOD) / 60 ))
  " 2>/dev/null || echo "?")

  if [[ "$STALE" != "?" && "$STALE" -ge "$STALE_MINS" ]]; then
    echo "  WARNING: log not updated in ${STALE}m (threshold: ${STALE_MINS}m) — may be hung"
  fi

  echo "  --- last $TAIL_LINES lines of $LOG_FILE ---"
  $SSH "tail -$TAIL_LINES '$LOG_FILE'" 2>/dev/null || echo "  (could not read log)"
fi
