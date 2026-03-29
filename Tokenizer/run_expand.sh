#!/usr/bin/env bash
# Run expand_nemotron_bhashakritika.py detached so SSH logout does not stop it.
# Progress: tail -f logs/latest.log
# Status:   ./run_expand.sh status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/expand_nemotron_bhashakritika.py"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_FILE="${SCRIPT_DIR}/expand_nemotron.pid"
mkdir -p "$LOG_DIR"

# Optional Python venv (edit if you use a different path)
if [[ -f /opt/venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source /opt/venv/bin/activate
elif [[ -f "${SCRIPT_DIR}/../.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/../.venv/bin/activate"
fi

usage() {
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
  echo ""
  echo "Usage:"
  echo "  $0                  # start in background (nohup), pass-through args below"
  echo "  $0 fg [args...]     # run in foreground (no nohup)"
  echo "  $0 status           # show PID and whether process is alive"
  echo "  $0 logs             # print path to latest log"
  echo ""
  echo "Examples:"
  echo "  $0 --tokenizer-only"
  echo "  $0 --tokenizer-only --languages hindi,bengali"
  echo "  $0 fg --tokenizer-only"
  echo ""
  echo "After logout/login, check progress:"
  echo "  tail -f ${LOG_DIR}/latest.log"
  echo "  $0 status"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "status" ]]; then
  if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file at $PID_FILE (job not started with this script, or never run)."
    exit 1
  fi
  pid="$(cat "$PID_FILE")"
  if ps -p "$pid" -o pid,cmd --no-headers 2>/dev/null | grep -q .; then
    echo "Running PID $pid"
    ps -p "$pid" -o pid,etime,cmd
  else
    echo "Not running (stale PID $pid in $PID_FILE). See logs in $LOG_DIR"
  fi
  exit 0
fi

if [[ "${1:-}" == "logs" ]]; then
  if [[ -L "${LOG_DIR}/latest.log" || -f "${LOG_DIR}/latest.log" ]]; then
    echo "${LOG_DIR}/latest.log"
  else
    echo "No latest.log yet under $LOG_DIR"
    exit 1
  fi
  exit 0
fi

FOREGROUND=0
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "fg" || "$arg" == "foreground" ]]; then
    FOREGROUND=1
  else
    ARGS+=("$arg")
  fi
done

if [[ ! -f "$PY" ]]; then
  echo "Missing $PY" >&2
  exit 1
fi

cd "$SCRIPT_DIR"

if [[ "$FOREGROUND" -eq 1 ]]; then
  exec python3 "$PY" "${ARGS[@]}"
fi

LOG_FILE="${LOG_DIR}/expand_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$LOG_FILE" "${LOG_DIR}/latest.log"

nohup python3 "$PY" "${ARGS[@]}" >>"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"

echo "Started background job PID $(cat "$PID_FILE")"
echo "Log file:    $LOG_FILE"
echo "Also linked: ${LOG_DIR}/latest.log"
echo ""
echo "Monitor: tail -f ${LOG_DIR}/latest.log"
echo "Status:  ${SCRIPT_DIR}/run_expand.sh status"
echo ""
echo "Note: Some clusters kill long jobs on login nodes; if that happens, use Slurm (sbatch)."
