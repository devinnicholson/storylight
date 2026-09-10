#!/usr/bin/env bash
# Promote one shadow-accepted trained planner with automatic exact rollback.

set -euo pipefail
IFS=$'\n\t'
PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH
unset BASH_ENV ENV CDPATH GLOBIGNORE

readonly CONFIG_FILE="/etc/storylight/storylight.env"
readonly STATE_DIR="/var/lib/storylight-trusted/trained-planner"
readonly ACTIVE_STATE="$STATE_DIR/active.env"
readonly CANDIDATE_ROOT="/var/lib/storylight-trusted/trained-planner-candidates"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly INSTALLER="${STORYLIGHT_CANDIDATE_INSTALLER:-$SCRIPT_DIR/install-trained-planner-candidate.sh}"
readonly TERMINAL_RECORDER="$SCRIPT_DIR/record-trained-planner-terminal-evidence.py"
readonly ACCEPTED_ENGINE_SHA256="95b69991b68c57a2d2d4bfa4116feb9ec57295588551d109353a42a9c16c4fdf"
readonly CACHE_CONTRACT_REVISION="semantic-v18-tensorrt-slot-privacy"
target_user="${STORYLIGHT_SERVICE_USER:-${SUDO_USER:-}}"
candidate_id=""
manifest_sha256=""
gate_evidence=""
gate_evidence_sha256=""
approval_token=""
terminal_evidence_dir=""
dry_run=0

usage() {
  cat <<'EOF'
Usage: sudo promote-trained-planner.sh OPTIONS

Required:
  --user USER
  --candidate-id ID
  --manifest-sha256 SHA256
  --gate-evidence PATH         Passed orchestrator gate artifact.
  --gate-evidence-sha256 SHA256
  --approval-token TOKEN       Exact token printed by --dry-run.
  --terminal-evidence-directory PATH  New root-owned terminal artifact directory.

Options:
  --dry-run                    Verify and print the one-purpose token only.
  -h, --help                   Show this help.
EOF
}

while (($#)); do
  case "$1" in
    --user)
      [[ $# -ge 2 ]] || { printf '%s\n' '--user requires a value' >&2; exit 64; }
      target_user="$2"
      shift 2
      ;;
    --candidate-id)
      [[ $# -ge 2 ]] || { printf '%s\n' '--candidate-id requires a value' >&2; exit 64; }
      candidate_id="$2"
      shift 2
      ;;
    --manifest-sha256)
      [[ $# -ge 2 ]] || { printf '%s\n' '--manifest-sha256 requires a value' >&2; exit 64; }
      manifest_sha256="$2"
      shift 2
      ;;
    --gate-evidence)
      [[ $# -ge 2 ]] || { printf '%s\n' '--gate-evidence requires a value' >&2; exit 64; }
      gate_evidence="$2"
      shift 2
      ;;
    --gate-evidence-sha256)
      [[ $# -ge 2 ]] || { printf '%s\n' '--gate-evidence-sha256 requires a value' >&2; exit 64; }
      gate_evidence_sha256="$2"
      shift 2
      ;;
    --approval-token)
      [[ $# -ge 2 ]] || { printf '%s\n' '--approval-token requires a value' >&2; exit 64; }
      approval_token="$2"
      shift 2
      ;;
    --terminal-evidence-directory)
      [[ $# -ge 2 ]] || {
        printf '%s\n' '--terminal-evidence-directory requires a value' >&2
        exit 64
      }
      terminal_evidence_dir="$2"
      shift 2
      ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
done

if [[ ! "$target_user" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]] \
  || ! id "$target_user" >/dev/null 2>&1; then
  printf 'Select a valid Storylight service user with --user.\n' >&2
  exit 65
fi
if [[ ! "$candidate_id" =~ ^[a-z0-9][a-z0-9-]{2,95}$ ]] \
  || [[ ! "$manifest_sha256" =~ ^[a-f0-9]{64}$ ]] \
  || [[ ! "$gate_evidence_sha256" =~ ^[a-f0-9]{64}$ ]] \
  || [[ "$terminal_evidence_dir" != /var/lib/storylight-trusted/trained-planner/evidence/* ]] \
  || [[ "$(realpath -m -- "$terminal_evidence_dir")" != "$terminal_evidence_dir" ]] \
  || [[ "$(dirname -- "$terminal_evidence_dir")" != /var/lib/storylight-trusted/trained-planner/evidence ]] \
  || [[ -e "$terminal_evidence_dir" || -L "$terminal_evidence_dir" ]] \
  || [[ ! -x "$TERMINAL_RECORDER" ]]; then
  usage >&2
  exit 64
fi
readonly CANDIDATE_DIR="$CANDIDATE_ROOT/$candidate_id"
verification="$(
  "$INSTALLER" \
    --bundle "$CANDIDATE_DIR" \
    --expected-manifest-sha256 "$manifest_sha256" \
    --verify-only \
    --installed-layout
)"
model_revision="$(printf '%s\n' "$verification" | sed -n 's/^model_revision=//p')"
engine_sha256="$(printf '%s\n' "$verification" | sed -n 's/^engine_sha256=//p')"
if [[ ! "$model_revision" =~ ^sha256:[a-f0-9]{64}$ ]] \
  || [[ "$engine_sha256" != "${model_revision#sha256:}" ]]; then
  printf 'Candidate verification did not return one exact engine identity.\n' >&2
  exit 65
fi
if [[ ! -f "$gate_evidence" || -L "$gate_evidence" ]]; then
  printf 'Gate evidence must be a regular file, not a symbolic link.\n' >&2
  exit 65
fi
gate_mode="$(stat -c '%a' "$gate_evidence")"
case "$gate_mode" in
  400|440|444|600|640|644) ;;
  *) printf 'Gate evidence has an unsafe file mode: %s.\n' "$gate_mode" >&2; exit 65 ;;
esac
if [[ "$(sha256sum "$gate_evidence" | cut -d' ' -f1)" != "$gate_evidence_sha256" ]]; then
  printf 'Gate evidence checksum differs from --gate-evidence-sha256.\n' >&2
  exit 65
fi
python3 - \
  "$gate_evidence" \
  "$CANDIDATE_DIR/candidate.manifest.json" \
  "$manifest_sha256" \
  "$candidate_id" \
  "$model_revision" \
  "$engine_sha256" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
candidate_manifest_sha256 = sys.argv[3]
candidate_id = sys.argv[4]
model_revision = sys.argv[5]
engine_sha256 = sys.argv[6]


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON repeats key: {key}")
        result[key] = value
    return result


try:
    document = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
    )
    candidate_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
    )
except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
    raise SystemExit(f"Gate evidence is not valid UTF-8 JSON: {error}") from error
if not isinstance(document, dict) or not isinstance(candidate_manifest, dict):
    raise SystemExit("Gate evidence and candidate manifest must contain JSON objects.")
required_top_level = {
    "schema_version",
    "stage",
    "producer",
    "run_id",
    "training_run_id",
    "config_sha256",
    "dataset_manifest_sha256",
    "status",
    "inputs",
    "candidate_manifest_sha256",
    "candidate_identity",
    "baseline_identity",
    "hidden_custody_receipt_sha256",
    "evidence_sha256",
    "decision",
}
if set(document) != required_top_level:
    raise SystemExit("Gate evidence has an unexpected top-level contract.")
if (
    document.get("schema_version") != "1.0"
    or document.get("stage") != "gate"
    or document.get("producer") != "storylight-fidelity-gate-builder"
    or document.get("status") != "passed"
):
    raise SystemExit("Gate evidence must be a passed gate artifact.")
if document.get("candidate_manifest_sha256") != candidate_manifest_sha256:
    raise SystemExit("Gate evidence identifies another candidate manifest.")
if document.get("training_run_id") != candidate_manifest.get("training_run_id"):
    raise SystemExit("Gate training run ID differs from the installed candidate lineage.")
if document.get("config_sha256") != candidate_manifest.get("source_config_sha256"):
    raise SystemExit("Gate config checksum differs from the installed candidate lineage.")
if document.get("dataset_manifest_sha256") != candidate_manifest.get(
    "source_dataset_manifest_sha256"
):
    raise SystemExit("Gate dataset checksum differs from the installed candidate lineage.")
if not isinstance(document.get("inputs"), dict) or set(document["inputs"]) != {
    "jetson-shadow"
}:
    raise SystemExit("Gate input must bind exactly one Jetson shadow artifact.")
sha_fields = [
    document.get("config_sha256"),
    document.get("dataset_manifest_sha256"),
    document.get("hidden_custody_receipt_sha256"),
    document["inputs"].get("jetson-shadow"),
]
if any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in sha_fields):
    raise SystemExit("Gate evidence contains an invalid lineage SHA-256.")
identity_fields = {
    "candidate_id",
    "candidate_manifest_sha256",
    "engine_sha256",
    "model_revision",
}
candidate_identity = document.get("candidate_identity")
expected_candidate_identity = {
    "candidate_id": candidate_id,
    "candidate_manifest_sha256": candidate_manifest_sha256,
    "engine_sha256": engine_sha256,
    "model_revision": model_revision,
}
if candidate_identity != expected_candidate_identity:
    raise SystemExit("Gate candidate identity differs from the installed engine.")
baseline_identity = document.get("baseline_identity")
if not isinstance(baseline_identity, dict) or set(baseline_identity) != identity_fields:
    raise SystemExit("Gate baseline identity is incomplete.")
if not str(baseline_identity.get("candidate_id", "")).startswith("accepted-baseline-"):
    raise SystemExit("Gate baseline identity is not an accepted baseline snapshot.")
for field in ("candidate_manifest_sha256", "engine_sha256"):
    if re.fullmatch(r"[0-9a-f]{64}", str(baseline_identity.get(field, ""))) is None:
        raise SystemExit("Gate baseline identity has an invalid checksum.")
if baseline_identity.get("model_revision") != f"sha256:{baseline_identity['engine_sha256']}":
    raise SystemExit("Gate baseline model revision is not its engine checksum.")
decision = document.get("decision")
if not isinstance(decision, dict) or set(decision) != {"passed", "reasons", "checks"}:
    raise SystemExit("Gate evidence has no typed promotion decision.")
checks = decision.get("checks")
if (
    decision.get("passed") is not True
    or decision.get("reasons") != []
    or not isinstance(checks, dict)
    or not checks
    or any(type(value) is not bool or not value for value in checks.values())
):
    raise SystemExit("Gate evidence promotion decision did not pass cleanly.")
evidence = document.get("evidence_sha256")
required = {
    "baseline_development_summary",
    "baseline_hidden_summary",
    "candidate_development_summary",
    "candidate_hidden_summary",
    "candidate_manifest",
    "contest",
    "dataset_manifest",
    "human_review",
    "runtime",
}
if not isinstance(evidence, dict) or set(evidence) != required:
    raise SystemExit("Gate evidence bindings are incomplete.")
if any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in evidence.values()):
    raise SystemExit("Gate evidence contains an invalid evidence SHA-256.")
if evidence["candidate_manifest"] != candidate_manifest_sha256:
    raise SystemExit("Gate candidate manifest evidence is not the installed manifest.")
if evidence["dataset_manifest"] != document["dataset_manifest_sha256"]:
    raise SystemExit("Gate dataset evidence is not the candidate dataset.")
PY
readonly EXPECTED_APPROVAL_TOKEN="PROMOTE_STORYLIGHT_TRAINED_PLANNER:${target_user}:${candidate_id}:${manifest_sha256}:${gate_evidence_sha256}:${terminal_evidence_dir}"

if ((dry_run == 1)); then
  printf '%s\n' "$verification"
  printf 'Required approval token: %s\n' "$EXPECTED_APPROVAL_TOKEN"
  printf 'terminal_evidence_directory=%s\n' "$terminal_evidence_dir"
  printf 'Would atomically back up %s, route the trained model on 11435, and retain exact rollback.\n' \
    "$CONFIG_FILE"
  printf 'Dry run complete; no files or services changed.\n'
  exit 0
fi
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  printf 'Run candidate promotion with sudo.\n' >&2
  exit 64
fi
if [[ "$approval_token" != "$EXPECTED_APPROVAL_TOKEN" ]]; then
  printf 'Promotion requires the exact one-purpose approval token printed by --dry-run.\n' >&2
  exit 77
fi
if [[ ! -f "$CONFIG_FILE" || -L "$CONFIG_FILE" ]] \
  || [[ "$(stat -c '%U:%G:%a' "$CONFIG_FILE")" != "root:root:600" ]]; then
  printf 'Storylight environment is missing or unsafe.\n' >&2
  exit 78
fi
if [[ -e "$ACTIVE_STATE" ]]; then
  printf 'A trained planner promotion state already exists; rollback or reconcile it first.\n' >&2
  exit 78
fi
if swapon --show --noheadings | grep -q .; then
  printf 'Promotion refuses active swap. Disable build-only swap first.\n' >&2
  exit 70
fi
if ! grep -Fxq 'STORYLIGHT_LIVE_SCENE_PLANNER_BACKEND=tensorrt_slots' "$CONFIG_FILE" \
  || ! grep -Fxq 'STORYLIGHT_LIVE_SCENE_PLANNER_BASE_URL=http://127.0.0.1:11435' "$CONFIG_FILE" \
  || ! grep -Fxq "STORYLIGHT_LIVE_SCENE_PLANNER_MODEL_REVISION=sha256:${ACCEPTED_ENGINE_SHA256}" \
    "$CONFIG_FILE"; then
  printf 'Production is not the exact accepted TensorRT configuration.\n' >&2
  exit 78
fi

target_uid="$(id -u "$target_user")"
target_home="$(getent passwd "$target_user" | cut -d: -f6)"
if [[ -z "$target_home" || "$target_home" != /* || ! -d "$target_home" ]]; then
  printf 'Cannot resolve the Storylight service user home directory.\n' >&2
  exit 65
fi
readonly TARGET_UID="$target_uid"
readonly TARGET_HOME="$target_home"
readonly ACCEPTED_ENGINE="$TARGET_HOME/.local/share/storylight/tensorrt-edgellm-v0.10.0/models/gemma4-e2b-it-int4-awq-v010/engines/llm/llm.engine"
if [[ ! -s "$ACCEPTED_ENGINE" || -L "$ACCEPTED_ENGINE" ]] \
  || [[ "$(sha256sum "$ACCEPTED_ENGINE" | cut -d' ' -f1)" != "$ACCEPTED_ENGINE_SHA256" ]]; then
  printf 'The exact accepted TensorRT engine is missing or changed.\n' >&2
  exit 78
fi

user_systemctl() {
  runuser -u "$target_user" -- env \
    XDG_RUNTIME_DIR="/run/user/${TARGET_UID}" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/${TARGET_UID}/bus" \
    systemctl --user "$@"
}

readonly UNIT="storylight-trained-planner-candidate@${candidate_id}.service"
readonly ACCEPTED_UNIT="storylight-tensorrt-planner.service"
readonly GEMMA_UNIT="storylight-gemma.service"
readonly KIOSK_UNIT="storylight-kiosk.service"
readonly TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly BACKUP_DIR="$STATE_DIR/rollback/${TIMESTAMP}-${candidate_id}"
readonly BACKUP_FILE="$BACKUP_DIR/storylight.env"
readonly PORT_ENV="/etc/storylight/trained-planner/${candidate_id}.env"
readonly ACTIVE_MODEL_REVISION="$model_revision"
kiosk_was_active=0
promotion_complete=0

write_state() {
  phase="$1"
  temporary="$(mktemp "$STATE_DIR/.active.env.XXXXXX")"
  {
    printf 'PHASE=%s\n' "$phase"
    printf 'CANDIDATE_ID=%s\n' "$candidate_id"
    printf 'MANIFEST_SHA256=%s\n' "$manifest_sha256"
    printf 'GATE_EVIDENCE_SHA256=%s\n' "$gate_evidence_sha256"
    printf 'GATE_PRODUCER=%s\n' 'storylight-fidelity-gate-builder'
    printf 'MODEL_REVISION=%s\n' "$model_revision"
    printf 'ENGINE_SHA256=%s\n' "$engine_sha256"
    printf 'CACHE_CONTRACT_REVISION=%s\n' "$CACHE_CONTRACT_REVISION"
    printf 'BACKUP_FILE=%s\n' "$BACKUP_FILE"
    printf 'BACKUP_SHA256=%s\n' "$(sha256sum "$BACKUP_FILE" | cut -d' ' -f1)"
    printf 'ACCEPTED_ENGINE_SHA256=%s\n' "$ACCEPTED_ENGINE_SHA256"
    printf 'KIOSK_WAS_ACTIVE=%s\n' "$kiosk_was_active"
  } >"$temporary"
  chown root:root "$temporary"
  chmod 0600 "$temporary"
  mv -f "$temporary" "$ACTIVE_STATE"
}

restore_exact_runtime() {
  exit_code=$?
  trap - EXIT INT TERM
  if ((promotion_complete == 0)); then
    printf 'Candidate promotion failed; restoring the exact accepted engine and environment.\n' >&2
    if ((kiosk_was_active == 1)); then
      user_systemctl stop "$KIOSK_UNIT" || true
    fi
    if [[ -s "$BACKUP_FILE" ]]; then
      restore_tmp="$(mktemp /etc/storylight/.storylight.env.rollback.XXXXXX)"
      install -o root -g root -m 0600 "$BACKUP_FILE" "$restore_tmp"
      mv -f "$restore_tmp" "$CONFIG_FILE"
    fi
    user_systemctl disable "$UNIT" >/dev/null 2>&1 || true
    user_systemctl enable "$ACCEPTED_UNIT" >/dev/null 2>&1 || true
    user_systemctl stop "$UNIT" || true
    if [[ -e "$PORT_ENV" ]]; then
      mv -f "$PORT_ENV" "$BACKUP_DIR/failed-port.env"
    fi
    user_systemctl stop "$GEMMA_UNIT" || true
    user_systemctl start "$ACCEPTED_UNIT" || true
    systemctl restart "storylight@${target_user}.service" \
      "storylight-controller@${target_user}.service" || true
    if ((kiosk_was_active == 1)); then
      user_systemctl start "$KIOSK_UNIT" || true
    fi
    if [[ -e "$ACTIVE_STATE" ]]; then
      install -d -o root -g root -m 0700 "$STATE_DIR/history"
      mv -f "$ACTIVE_STATE" "$STATE_DIR/history/${TIMESTAMP}-${candidate_id}-failed.env"
    fi
  fi
  exit "$exit_code"
}

install -d -o root -g root -m 0700 "$STATE_DIR" "$STATE_DIR/rollback" "$BACKUP_DIR"
install -d -o root -g root -m 0755 /etc/storylight/trained-planner
install -o root -g root -m 0600 "$CONFIG_FILE" "$BACKUP_FILE"
if user_systemctl is-active --quiet "$KIOSK_UNIT"; then
  kiosk_was_active=1
fi
write_state PROMOTING
trap restore_exact_runtime EXIT INT TERM

config_tmp="$(mktemp /etc/storylight/.storylight.env.candidate.XXXXXX)"
python3 - "$CONFIG_FILE" "$config_tmp" "$ACTIVE_MODEL_REVISION" \
  "$CACHE_CONTRACT_REVISION" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
revision = sys.argv[3]
contract_revision = sys.argv[4]
replacements = {
    "STORYLIGHT_LIVE_SCENE_PLANNER": "model",
    "STORYLIGHT_LIVE_SCENE_PLANNER_BACKEND": "tensorrt_slots",
    "STORYLIGHT_LIVE_SCENE_PLANNER_BASE_URL": "http://127.0.0.1:11435",
    "STORYLIGHT_LIVE_SCENE_PLANNER_MODEL_NAME": "llm",
    "STORYLIGHT_LIVE_SCENE_PLANNER_MAX_OUTPUT_TOKENS": "64",
    "STORYLIGHT_LIVE_SCENE_PLANNER_MODEL_REVISION": revision,
    "STORYLIGHT_LIVE_SCENE_PLANNER_CACHE_CONTRACT_REVISION": contract_revision,
    "STORYLIGHT_LIVE_SCENE_PLANNER_COMPACT_WIRE": "false",
}
remaining = dict(replacements)
updated = []
for line in source.read_text(encoding="utf-8").splitlines():
    name, separator, _ = line.partition("=")
    if separator and name in remaining:
        updated.append(f"{name}={remaining.pop(name)}")
    else:
        updated.append(line)
updated.extend(f"{name}={value}" for name, value in remaining.items())
destination.write_text("\n".join(updated) + "\n", encoding="utf-8")
PY
chown root:root "$config_tmp"
chmod 0600 "$config_tmp"
mv -f "$config_tmp" "$CONFIG_FILE"

port_tmp="$(mktemp /etc/storylight/trained-planner/.candidate.env.XXXXXX)"
printf 'STORYLIGHT_EDGELLM_SERVER_PORT=11435\n' >"$port_tmp"
chown root:root "$port_tmp"
chmod 0644 "$port_tmp"
mv -f "$port_tmp" "$PORT_ENV"

user_systemctl daemon-reload
if ((kiosk_was_active == 1)); then
  user_systemctl stop "$KIOSK_UNIT"
fi
user_systemctl stop "$ACCEPTED_UNIT"
user_systemctl stop "$GEMMA_UNIT" || true
for _ in $(seq 1 30); do
  if ! pgrep -u "$TARGET_UID" -f 'experimental.server|ollama serve|firefox-storylight' \
    >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
available_kib="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
if [[ ! "$available_kib" =~ ^[0-9]+$ ]] || ((available_kib < 4194304)); then
  printf 'Candidate startup requires at least 4 GiB available memory.\n' >&2
  exit 70
fi

user_systemctl reset-failed "$UNIT" || true
user_systemctl start "$UNIT"
for _ in $(seq 1 90); do
  if curl --fail --silent --max-time 1 http://127.0.0.1:11435/v1/models >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl --fail --silent --show-error http://127.0.0.1:11435/v1/models >/dev/null
systemctl restart "storylight@${target_user}.service" \
  "storylight-controller@${target_user}.service"
for _ in $(seq 1 30); do
  if curl --fail --silent --max-time 1 http://127.0.0.1:8080/readyz >/dev/null 2>&1 \
    && curl --fail --silent --max-time 1 http://127.0.0.1:8081/healthz >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl --fail --silent --show-error http://127.0.0.1:8080/readyz >/dev/null
curl --fail --silent --show-error http://127.0.0.1:8081/healthz >/dev/null
curl --fail --silent --show-error -X POST \
  -H 'content-type: application/json' \
  --data '{"text":"A copper fox raises a blue lantern and paper stars cross the quiet observatory.","visual_style":"layered paper theater","seed":20260901}' \
  http://127.0.0.1:8080/v1/live-scene-planner/prepare \
  >"$BACKUP_DIR/promotion-smoke.json"
python3 - "$BACKUP_DIR/promotion-smoke.json" "$model_revision" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    response = json.load(stream)
if response.get("revision") != sys.argv[2]:
    raise SystemExit("Promotion smoke response advertised the wrong model revision.")
if not 0 <= response.get("output_tokens", 65) <= 64:
    raise SystemExit("Promotion smoke response exceeded the bounded output contract.")
PY
if ((kiosk_was_active == 1)); then
  user_systemctl start "$KIOSK_UNIT"
  user_systemctl is-active --quiet "$KIOSK_UNIT"
fi
if swapon --show --noheadings | grep -q .; then
  printf 'Swap became active during candidate promotion.\n' >&2
  exit 70
fi
user_systemctl disable "$ACCEPTED_UNIT" >/dev/null
user_systemctl enable "$UNIT" >/dev/null
write_state ACTIVE
"$TERMINAL_RECORDER" \
  --outcome promoted \
  --user "$target_user" \
  --gate-artifact "$gate_evidence" \
  --gate-artifact-sha256 "$gate_evidence_sha256" \
  --candidate-manifest "$CANDIDATE_DIR/candidate.manifest.json" \
  --candidate-manifest-sha256 "$manifest_sha256" \
  --active-engine "$CANDIDATE_DIR/engines/llm/llm.engine" \
  --active-engine-sha256 "$engine_sha256" \
  --baseline-engine-sha256 "$ACCEPTED_ENGINE_SHA256" \
  --backup-config "$BACKUP_FILE" \
  --one-purpose-approval-token "$EXPECTED_APPROVAL_TOKEN" \
  --output-directory "$terminal_evidence_dir"
promotion_complete=1
trap - EXIT INT TERM
printf 'Trained planner %s is active with exact rollback at %s.\n' \
  "$candidate_id" "$BACKUP_FILE"
