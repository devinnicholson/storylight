"""Run the frozen evaluator over separately produced, content-addressed predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .commands import shell_join
from .configuration import load_config
from .integrity import sha256_file
from .runtime import approval_token, require_approval, run_checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--records-sha256", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--predictions-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--surface",
        choices=("raw", "postprocessed", "renderer-safe"),
        required=True,
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if sha256_file(args.records) != args.records_sha256:
        raise SystemExit("evaluation records SHA-256 mismatch")
    if sha256_file(args.predictions) != args.predictions_sha256:
        raise SystemExit("prediction JSONL SHA-256 mismatch")
    command = [
        "python3",
        "-m",
        "bookforge.fidelity_benchmark",
        "--records",
        str(args.records),
        "--predictions",
        str(args.predictions),
        "--output",
        str(args.output),
        "--surface",
        args.surface,
    ]
    run_id = f"evaluate-{args.surface}-{config.sha256[:12]}-{args.predictions_sha256[:12]}"
    token = approval_token(
        stage="evaluate",
        run_id=run_id,
        config_sha256=config.sha256,
        input_sha256=args.predictions_sha256,
    )
    print(
        json.dumps(
            {"run_id": run_id, "command": shell_join(command), "approval_token": token},
            indent=2,
            sort_keys=True,
        )
    )
    if not args.execute:
        return
    require_approval(token)
    run_checked(command, cwd=Path.cwd())


if __name__ == "__main__":
    main()
