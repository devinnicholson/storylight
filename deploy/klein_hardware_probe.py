"""Ten fixed renders comparing one GPU with frozen Klein image references."""

import hashlib
import json

SCHEDULE = (("warmup", 0), ("warmup", 1)) + tuple(("measured", index % 2) for index in range(8))
ERROR_TYPES = {
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "AttributeError",
    "RuntimeError",
    "MemoryError",
    "OutOfMemoryError",
    "OSError",
}


def require(condition):
    if not condition:
        raise ValueError("hardware probe validation failed")


def run_comparison(runtime, cases):
    records, ordinal, stage = [], None, "setup"

    def event(state, error=None):
        value = {"ordinal": ordinal, "stage": stage, "state": state}
        if error is not None:
            name = type(error).__name__
            value["exception_type"] = name if name in ERROR_TYPES else "OtherError"
        print(json.dumps({"hardware_progress": value}), flush=True)

    try:
        require(isinstance(cases, list) and len(cases) == 2)
        require([case["expected_bucket"] for case in cases] == [128, 256])
        for ordinal, (phase, case_index) in enumerate(SCHEDULE):
            case, stage = cases[case_index], "render"
            event("start")
            metrics, master, depth = runtime.render(case["prompt"], case["seed"])
            # Retain returned evidence even if its metadata later fails validation.
            records.append(
                {
                    "ordinal": ordinal,
                    "phase": phase,
                    "case_index": case_index,
                    "metrics": metrics,
                    "master": master,
                    "depth": depth,
                    "historical_images_exact": False,
                }
            )
            stage = "verification"
            require(
                metrics["seed"] == case["seed"]
                and metrics["sequence_bucket"] == case["expected_bucket"]
            )
            for name, data in (("master", master), ("depth", depth)):
                require(type(data) is bytes and len(data) > 0)
                require(hashlib.sha256(data).hexdigest() == metrics[f"{name}_sha256"])
            records[-1]["historical_images_exact"] = all(
                metrics[f"{name}_sha256"] == case[f"{name}_sha256"] for name in ("master", "depth")
            )
            event("verified")
    except Exception as error:
        event("failed", error)
        return {"status": "failed", "failure_stage": stage, "records": records}
    return {"status": "complete", "failure_stage": None, "records": records}
