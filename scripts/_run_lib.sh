# Shared helpers for the run_* stage scripts. Caller sets $LOG.
have () { [ -f "$1" ]; }
ts () { echo "=== $1 $(date) ===" | tee -a "$LOG"; }
run () {
  local label="$1"; shift
  ts "run $label"
  if ! "$@" >> "$LOG" 2>&1; then ts "ABORT $label exit=$?"; exit 1; fi
  ts "done $label"
}
