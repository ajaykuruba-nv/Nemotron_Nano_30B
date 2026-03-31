#!/usr/bin/env bash
# Run continued_bpe.py detached so SSH logout does not stop it.
# Progress: tail -f logs/latest_continued_bpe.log
# Status:   ./run_continued_bpe.sh status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/continued_bpe.py"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_FILE="${SCRIPT_DIR}/continued_bpe.pid"
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
  sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
  echo ""
  echo "Usage:"
  echo "  $0 [args...]        # start in background (nohup)"
  echo "  $0 fg [args...]     # run in foreground (no nohup)"
  echo "  $0 status           # show PID and whether process is alive"
  echo "  $0 logs             # print path to latest log"
  echo ""
  echo "Notes:"
  echo "  - If you don't pass --out-dir, continued_bpe.py will create:"
  echo "      ${SCRIPT_DIR}/outputs/continued_bpe_YYYYMMDD_HHMMSS"
  echo "  - If you don't pass --languages, defaults are used."
  echo ""
  echo "Examples:"
  echo "  $0 --tokenizer-only"
  echo "  $0 --tokenizer-only --samples-per-lang 1000"
  echo "  $0 --out-dir ${SCRIPT_DIR}/outputs/my_run --tokenizer-only"
  echo "  $0 fg --tokenizer-only --samples-per-lang 1000"
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
  if [[ -L "${LOG_DIR}/latest_continued_bpe.log" || -f "${LOG_DIR}/latest_continued_bpe.log" ]]; then
    echo "${LOG_DIR}/latest_continued_bpe.log"
  else
    echo "No latest log yet under $LOG_DIR"
    exit 1
  fi
  exit 0
fi

FOREGROUND=0
ARGS=()
HAS_LANGUAGES=0

for arg in "$@"; do
  if [[ "$arg" == "fg" || "$arg" == "foreground" ]]; then
    FOREGROUND=1
  else
    if [[ "$arg" == --languages* ]]; then
      HAS_LANGUAGES=1
    fi
    ARGS+=("$arg")
  fi
done

# If the user didn't explicitly provide the --languages flag,
# inject a safe default with no spaces to prevent bash-splitting bugs.
if [[ "$HAS_LANGUAGES" -eq 0 ]]; then
  echo "No --languages flag detected. Injecting safe default: hindi,bengali,tamil,telugu"
  ARGS+=("--languages" "hindi,bengali,tamil,telugu")
fi

if [[ ! -f "$PY" ]]; then
  echo "Missing $PY" >&2
  exit 1
fi

cd "$SCRIPT_DIR"

if [[ "$FOREGROUND" -eq 1 ]]; then
  echo "Running command: python3 $PY ${ARGS[*]}"
  exec python3 "$PY" "${ARGS[@]}"
fi

LOG_FILE="${LOG_DIR}/continued_bpe_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$LOG_FILE" "${LOG_DIR}/latest_continued_bpe.log"

echo "Running command: python3 $PY ${ARGS[*]}" >> "$LOG_FILE"
nohup python3 "$PY" "${ARGS[@]}" >>"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"

echo "Started background job PID $(cat "$PID_FILE")"
echo "Log file:    $LOG_FILE"
echo "Also linked: ${LOG_DIR}/latest_continued_bpe.log"
echo ""
echo "Monitor: tail -f ${LOG_DIR}/latest_continued_bpe.log"
echo "Status:  ${SCRIPT_DIR}/run_continued_bpe.sh status"

