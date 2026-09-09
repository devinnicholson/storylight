#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 && "$1" =~ ^[0-9a-f]{64}$ ]] || exit 2
cd /opt/bookforge-scene-adapter-v5
[[ ! -e cache-confirmation-01 && ! -L cache-confirmation-01 ]] || exit 2
printf '%s  %s\n' "$1" cache-bridge-compiled-01/completed.json | sha256sum --check --status
sha256sum --check --status <<'PINS'
1c3f0ecb5111fec4f4756f4f776af2ef5977d2cf222330414e3bc6c3d8132fd3  comparison/cache-confirmation/run.py
1a2579008b49f64d2a2d45f00a2c4ed5c63c0621d1ca6accbf0637ff4f79bd69  comparison/cache-confirmation/protocol.json
4ffc9adc3c15fbb6e684197a05b4b8bc2281f616ea02888f38989fff3acf58fb  comparison/cache-confirmation/inputs.jsonl
e42f6ca4a24d60a756c29122ea323b151ae9d89b4631a6fbba63e64adfd31a31  /tmp/bookforge-v5-cache-bridge-reuse.py
PINS
.venv/bin/python - <<'PY'
import json
from pathlib import Path
previous = json.loads(Path('cache-bridge-compiled-01/completed.json').read_text())
assert previous['calls'] == 54
assert previous['all_tokens_identical'] is True
assert previous['all_calls_token_identical'] is True
PY
exec /usr/bin/timeout --signal=TERM --kill-after=30s 1660s \
  .venv/bin/python comparison/cache-confirmation/run.py \
  --confirmation-dir comparison/cache-confirmation \
  --compiler-cache-root /tmp/bookforge-cache-confirmation-compiler-01 \
  --bridge-helper /tmp/bookforge-v5-cache-bridge-reuse.py \
  --v5-trainer payload/train.py --v5-runner comparison/run.py \
  --v2-trainer payload/v2-train.py --v3-runner comparison/v3-run.py \
  --merge-helper comparison/serving-profile.py \
  --model-dir model --model-manifest model-manifest.json \
  --model-manifest-sha256 703bbb89d61aaed083846d7cb3d4ee1a68220e25de93a035b1f4b49d24062f2d \
  --training-run training-01 \
  --completed-sha256 66326bd05dd934feae4ec1375a7b004f6efb0d267c0e0be72bfd3881f781768c \
  --data-dir payload/data --grammar comparison/grammar.ebnf \
  --output cache-confirmation-01 --max-runtime-seconds 1600
