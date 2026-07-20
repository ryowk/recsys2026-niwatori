# Shared helpers for the run_* stage scripts. Caller sets $LOG.
have () { [ -f "$1" ]; }
artifact_complete () {
  local dir="$1"; shift
  [ -f "$dir/manifest.json" ] || return 1
  local name
  for name in "$@"; do
    [ -f "$dir/$name" ] || return 1
  done
}
split_artifact_complete () {
  artifact_complete "$1" sessions.jsonl rows.jsonl
}
candidate_artifact_complete () {
  artifact_complete "$1" candidates.npz turns.jsonl
}
union_artifact_complete () {
  artifact_complete "$1" candidates.npz turns.jsonl source_features.npz
}
ranked_artifact_complete () {
  artifact_complete "$1" ranked.npz turns.jsonl model.txt
}
ts () { echo "=== $1 $(date) ===" | tee -a "$LOG"; }
run () {
  local label="$1"; shift
  ts "run $label"
  if "$@" >> "$LOG" 2>&1; then
    ts "done $label"
    return
  else
    local status=$?
    ts "ABORT $label exit=$status"
    exit "$status"
  fi
}
