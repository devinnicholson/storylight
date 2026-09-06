import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.snapshot_supervision import SnapshotLogGuard  # noqa: E402


def test_snapshot_log_failure_is_sticky_and_log_loss_fails_closed(tmp_path):
    log = tmp_path / "platform.log"
    log.write_text("Snapshot created. Restoring Function from memory snapshot.\n")
    guard = SnapshotLogGuard(log)
    assert guard.check() is None
    with log.open("a") as stream:
        stream.write("Failed creating Function memory snap")
    assert guard.check() is None
    with log.open("a") as stream:
        stream.write("shot.\n")
    assert guard.check() == "snapshot_platform_failure"
    log.write_text("Restoring Function from memory snapshot.\n")
    assert guard.check() == "snapshot_platform_failure"

    guard = SnapshotLogGuard(log)
    assert guard.check() is None
    log.write_text("")
    assert guard.check() == "platform_log_truncated"
    assert SnapshotLogGuard(tmp_path / "missing").check() == "platform_log_unavailable"
    log.write_text("x" * 65)
    assert SnapshotLogGuard(log, maximum_bytes=64).check() == "platform_log_exceeded_bound"

    for message, expected in (
        ("Runner failed with exception: device error", "snapshot_worker_failure"),
        ("Retrying task without snapshot", "snapshot_platform_retry"),
        ("ValueError: warmed snapshot capture allowance exhausted", "snapshot_platform_failure"),
    ):
        log.write_text(message)
        assert SnapshotLogGuard(log).check() == expected
