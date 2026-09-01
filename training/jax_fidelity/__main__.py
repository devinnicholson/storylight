"""Small local dispatcher; model-heavy work remains opt-in in each entrypoint."""

from __future__ import annotations

import argparse
import json

from .configuration import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    print(
        json.dumps(
            {
                "valid": True,
                "experiment_id": config.experiment_id,
                "config_sha256": config.sha256,
                "model_revision": config.production["model_revision"],
                "maxtext_revision": config.versions["maxtext_revision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
