"""Launch one MaxText module after explicit JAX cache configuration."""

from __future__ import annotations

import json
import runpy
import sys

from .compilation_cache import (
    configure_persistent_compilation_cache,
    write_cache_receipt,
)


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].startswith("maxtext."):
        raise SystemExit("an explicit maxtext.* module is required")
    target_module = sys.argv[1]

    import jax

    receipt = configure_persistent_compilation_cache(jax)
    if receipt.get("configured") is True:
        cache_override = f"jax_cache_dir={receipt['cache_directory']}"
        if cache_override not in sys.argv[2:]:
            raise RuntimeError(
                "MaxText must receive the trusted JAX cache directory as an "
                "explicit config override"
            )
        if "dump_hlo=false" not in sys.argv[2:]:
            raise RuntimeError("MaxText HLO dumping must remain disabled for persistent caching")
    receipt_path = write_cache_receipt(receipt)
    print(
        "Bookforge JAX cache: "
        + json.dumps(
            {**receipt, "receipt_path": str(receipt_path) if receipt_path else None},
            sort_keys=True,
        ),
        flush=True,
    )
    sys.argv = [target_module, *sys.argv[2:]]
    runpy.run_module(target_module, run_name="__main__", alter_sys=False)


if __name__ == "__main__":
    main()
