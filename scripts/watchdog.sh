#!/usr/bin/env bash
# watchdog.sh — watch a PID and restart it verbatim if it dies
#
# Usage:
#   watchdog.sh <pid> [options]
#
# Options:
#   -i INTERVAL     poll interval in seconds (default: 30)
#   -h HOST         SSH host (optional, for remote processes)
#   -p PORT         SSH port (default: 22)
#   -k KEY          SSH key file (default: ~/.ssh/id_rsa)
#   -u USER         SSH user (default: root)
#   -w WORKDIR      working directory to restore on restart (default: captured from /proc)
#
# Examples:
#   Local:   watchdog.sh 12345
#   Remote:  watchdog.sh 12345 -h 216.249.100.66 -p 13669 -k ~/.ssh/id_runpod_key

set -euo pipefail

PID="${1:-}"
if [[ -z "$PID" ]]; then
  echo "Usage: watchdog.sh <pid> [options]" >&2
  exit 1
fi
shift

INTERVAL=30
HOST=""
PORT=22
KEY=~/.ssh/id_rsa
USER=root
WORKDIR=""

while getopts "i:h:p:k:u:w:" opt; do
  case $opt in
    i) INTERVAL="$OPTARG" ;;
    h) HOST="$OPTARG" ;;
    p) PORT="$OPTARG" ;;
    k) KEY="$OPTARG" ;;
    u) USER="$OPTARG" ;;
    w) WORKDIR="$OPTARG" ;;
    *) echo "Unknown option: $opt" >&2; exit 1 ;;
  esac
done

# Build exec function — local or remote
if [[ -n "$HOST" ]]; then
  RUN() { ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$USER@$HOST" "$1"; }
else
  RUN() { bash -c "$1"; }
fi

# Capture process info from /proc while it's alive
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Attaching to PID $PID"

# Capture cmdline as a properly shell-quoted string (one %q token per arg)
CMDLINE=$(RUN "tr '\0' '\n' < /proc/$PID/cmdline | while IFS= read -r arg; do printf '%q ' \"\$arg\"; done" 2>/dev/null) || {
  echo "Error: PID $PID not found" >&2; exit 1
}
CMDLINE="${CMDLINE% }"  # strip trailing space

ENVIRON=$(RUN "cat /proc/$PID/environ | tr '\0' '\n'" 2>/dev/null || echo "")
if [[ -z "$WORKDIR" ]]; then
  WORKDIR=$(RUN "readlink /proc/$PID/cwd" 2>/dev/null || echo "")
fi

echo "  CMD: $CMDLINE"
echo "  CWD: $WORKDIR"

# Watch loop
while true; do
  sleep "$INTERVAL"

  if RUN "kill -0 $PID 2>/dev/null" 2>/dev/null; then
    continue
  fi

  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] PID $PID died — restarting..."

  ENV_EXPORTS=$(echo "$ENVIRON" | grep -v '^$' | sed "s/'/'\\\\''/g; s/^/export '/; s/$/'/")

  # Build restart script as newline-separated statements.
  # ENV_EXPORTS is multi-line (one 'export KEY=VAL' per line), so we cannot use
  # && chaining — a line starting with && is a shell syntax error. Newlines work
  # as statement separators and keep the structure valid regardless of how many
  # environment variables there are.
  RESTART_SCRIPT=$(printf 'cd %q\n%s\nnohup bash -c %q >/dev/null 2>&1 & echo $!' \
    "$WORKDIR" "$ENV_EXPORTS" "$CMDLINE")

  NEW_PID=$(RUN "$RESTART_SCRIPT" 2>/dev/null || echo "")

  if [[ -n "$NEW_PID" ]]; then
    echo "  Restarted (new PID $NEW_PID)"
    PID="$NEW_PID"
  else
    echo "  WARNING: restart failed, will retry next interval"
  fi
done
