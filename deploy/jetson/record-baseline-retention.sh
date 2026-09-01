#!/usr/bin/env bash
# Record an explicit no-promotion outcome after fresh Jetson health probes.

set -euo pipefail
IFS=$'\n\t'
PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly RECORDER="$SCRIPT_DIR/record-trained-planner-terminal-evidence.py"
target_user="${BOOKFORGE_SERVICE_USER:-${SUDO_USER:-}}"
gate_artifact=""
gate_sha256=""
candidate_manifest=""
manifest_sha256=""
baseline_engine=""
baseline_sha256=""
output_directory=""
approval_token=""
dry_run=0

usage() {
  cat <<'EOF'
Usage: sudo record-baseline-retention.sh OPTIONS

Required:
  --user USER
  --gate-artifact PATH
  --gate-artifact-sha256 SHA256
  --candidate-manifest PATH
  --candidate-manifest-sha256 SHA256
  --baseline-engine PATH
  --baseline-engine-sha256 SHA256
  --output-directory PATH
  --approval-token TOKEN       Exact token printed by --dry-run.

Options:
  --dry-run
EOF
}

while (($#)); do
  case "$1" in
    --user) target_user="$2"; shift 2 ;;
    --gate-artifact) gate_artifact="$2"; shift 2 ;;
    --gate-artifact-sha256) gate_sha256="$2"; shift 2 ;;
    --candidate-manifest) candidate_manifest="$2"; shift 2 ;;
    --candidate-manifest-sha256) manifest_sha256="$2"; shift 2 ;;
    --baseline-engine) baseline_engine="$2"; shift 2 ;;
    --baseline-engine-sha256) baseline_sha256="$2"; shift 2 ;;
    --output-directory) output_directory="$2"; shift 2 ;;
    --approval-token) approval_token="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
done

if [[ ! "$target_user" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]] \
  || ! id "$target_user" >/dev/null 2>&1 \
  || [[ ! "$gate_sha256" =~ ^[a-f0-9]{64}$ ]] \
  || [[ ! "$manifest_sha256" =~ ^[a-f0-9]{64}$ ]] \
  || [[ ! "$baseline_sha256" =~ ^[a-f0-9]{64}$ ]] \
  || [[ "$output_directory" != /var/lib/bookforge-trusted/trained-planner/evidence/* ]] \
  || [[ "$(realpath -m -- "$output_directory")" != "$output_directory" ]] \
  || [[ "$(dirname -- "$output_directory")" != /var/lib/bookforge-trusted/trained-planner/evidence ]] \
  || [[ ! -x "$RECORDER" ]]; then
  usage >&2
  exit 64
fi
for input in "$gate_artifact:$gate_sha256" "$candidate_manifest:$manifest_sha256" \
  "$baseline_engine:$baseline_sha256"; do
  path="${input%:*}"
  digest="${input##*:}"
  if [[ ! -f "$path" || -L "$path" ]] \
    || [[ "$(sha256sum "$path" | cut -d' ' -f1)" != "$digest" ]]; then
    printf 'Retention input is missing, linked, or changed: %s\n' "$path" >&2
    exit 65
  fi
done

action_sha256="$(
  printf '%s\0' "$target_user" "$gate_sha256" "$manifest_sha256" \
    "$baseline_sha256" "$output_directory" | sha256sum | cut -d' ' -f1
)"
readonly EXPECTED_APPROVAL_TOKEN="RETAIN_BOOKFORGE_ACCEPTED_BASELINE:${action_sha256}"
if ((dry_run == 1)); then
  printf 'Required approval token: %s\n' "$EXPECTED_APPROVAL_TOKEN"
  printf 'Would retain engine %s and atomically publish terminal evidence at %s.\n' \
    "$baseline_sha256" "$output_directory"
  exit 0
fi
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  printf 'Run baseline retention recording with sudo.\n' >&2
  exit 64
fi
if [[ "$approval_token" != "$EXPECTED_APPROVAL_TOKEN" ]]; then
  printf 'Retention requires the exact one-purpose token printed by --dry-run.\n' >&2
  exit 77
fi

exec "$RECORDER" \
  --outcome retained \
  --user "$target_user" \
  --gate-artifact "$gate_artifact" \
  --gate-artifact-sha256 "$gate_sha256" \
  --candidate-manifest "$candidate_manifest" \
  --candidate-manifest-sha256 "$manifest_sha256" \
  --active-engine "$baseline_engine" \
  --active-engine-sha256 "$baseline_sha256" \
  --baseline-engine-sha256 "$baseline_sha256" \
  --one-purpose-approval-token "$EXPECTED_APPROVAL_TOKEN" \
  --output-directory "$output_directory"
