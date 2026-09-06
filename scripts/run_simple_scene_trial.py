#!/usr/bin/env python3
"""Supervise one authorized simple-scene trial, including exact-app shutdown verification."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import suppress
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import benchmark_simple_scenes as harness  # noqa: E402

APP = "bookforge-klein-simple-scenes"
DEPLOYMENT = ROOT / "deploy/modal_klein_simple_scenes.py"
MANIFEST = ROOT / "benchmarks/simple-scenes-2026-09-05/manifest.json"
WORK_SECONDS, CLEANUP_SECONDS = 600, 60
require = harness.require


def write(path, value):
    harness.legacy.write_exclusive(path, harness.preparation.encoded(value))


def validate_metadata(data):
    require(len(data["ranked_functions"]) == 1)
    function = data["ranked_functions"][0]["function"]
    require(function["image_id"] == harness.cold.IMAGE_ID)
    require(
        function["resources"]
        == {
            "gpu_config": {"count": 1, "gpu_type": "L4"},
            "memory_mb": 65536,
            "memory_mb_max": 65536,
            "milli_cpu": 8000,
            "milli_cpu_max": 8000,
        }
    )
    scale = function["autoscaler_settings"]
    require(scale["max_containers"] == 1 and scale.get("min_containers", 0) == 0)
    require(scale.get("buffer_containers", 0) == 0 and scale["scaledown_window"] == 90)
    require(function["max_inputs"] == 1 and not function.get("single_use_containers", False))
    require(function["startup_timeout_secs"] == 120 and function["timeout_secs"] == 60)
    require(not function.get("cloud_provider_str") and function["routing_region"] == "us-east")
    require(function["scheduler_placement"]["regions"] == ["us"])
    require(function.get("retry_policy", {}).get("retries", 0) == 0)
    require(not function.get("checkpointing_enabled", False))
    require(not function.get("enable_gpu_snapshot", False))
    require(not function.get("_experimental_enable_gpu_snapshot", False))
    require(not function.get("experimental_options"))
    require(function.get("is_class") is True and not function.get("web_url"))
    keys = (
        "image_id",
        "resources",
        "autoscaler_settings",
        "max_inputs",
        "startup_timeout_secs",
        "timeout_secs",
        "routing_region",
        "scheduler_placement",
        "retry_policy",
        "checkpointing_enabled",
        "is_class",
    )
    return {key: function[key] for key in keys if key in function}


async def metadata(app_id):
    from google.protobuf.json_format import MessageToDict
    from modal.client import _Client
    from modal_proto import api_pb2

    require(re.fullmatch(r"ap-[A-Za-z0-9]+", app_id))
    client = await _Client.from_env()
    response = await client.stub.FunctionGet(
        api_pb2.FunctionGetRequest(
            app_name=APP,
            object_tag="SimpleSceneRenderer.*",
            environment_name="main",
        )
    )
    layout = await client.stub.AppGetLayout(api_pb2.AppGetLayoutRequest(app_id=app_id))
    require(response.function_id in layout.app_layout.function_ids.values())
    result = validate_metadata(MessageToDict(response.function, preserving_proto_field_name=True))
    return {**result, "app_id": app_id, "function_id": response.function_id}


def preflight(args):
    require(Path(args.python).absolute() == Path(sys.executable).absolute())
    require(os.environ.get("HF_HUB_OFFLINE") == "1")
    require(os.environ.get("TRANSFORMERS_OFFLINE") == "1")
    require(args.manifest.resolve() == MANIFEST and args.manifest.absolute() == MANIFEST)
    manifest = harness.read_manifest(args.manifest)
    authorization = harness.read_authorization(
        args.authorization,
        args.authorization_sha256,
        args.manifest,
    )
    require(manifest["status"] == "authorized" and time.time() < manifest["expires_at"])
    require(manifest["expires_at"] <= time.time() + 7200)
    require(args.ledger.resolve() == args.ledger.absolute() and not args.ledger.is_symlink())
    require(harness.legacy.file_hash(args.ledger) == authorization["ledger_sha256"])
    ledger = harness.legacy.protocol.decode_json(args.ledger.read_bytes())
    reservation = "reservation:simple-scenes-20260905-a"
    require(authorization["reservation_id"] == reservation)
    require(Decimal(str(ledger["reservations"][reservation])) == Decimal("1.75"))
    envelope = ledger["envelope"]
    usage = Decimal(str(ledger["estimated_usage_usd"]))
    require(usage <= Decimal(str(envelope["run_cap_usd"])))
    require(
        usage + Decimal(str(envelope["usage_before_lab_usd"]))
        <= sum(
            Decimal(str(envelope[key])) * sign
            for key, sign in (
                ("monthly_credit_usd", 1),
                ("authorized_paid_usd", 1),
                ("reserve_usd", -1),
            )
        )
    )
    tokens = harness.token_preflight()
    write(
        args.output / "preflight.json",
        {
            "manifest_sha256": authorization["manifest_sha256"],
            "token_preflight": tokens,
            "authorization_sha256": args.authorization_sha256,
            "ledger_sha256": authorization["ledger_sha256"],
            "cost_ceiling_usd": 1.75,
            "maximum_operations": 12,
            "maximum_initializations": 1,
            "generation_calls": 0,
        },
    )
    return manifest


def inventory_rows(raw):
    require(len(raw) <= 4_000_000)
    rows = json.loads(raw)
    require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows))
    return rows


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.deadline = self.started + WORK_SECONDS
        self.app_id = None
        self.previous_ids = set()
        self.deployment_started = False
        self.children = []
        self.log_process = None
        self.record = {
            "global_timeout_seconds": WORK_SECONDS,
            "shutdown_allowance_seconds": CLEANUP_SECONDS,
            "maximum_operations": 12,
            "maximum_initializations": 1,
            "cost_ceiling_usd": 1.75,
            "supervisor_sha256": harness.legacy.file_hash(Path(__file__)),
            "external_app_shutdown_verified": False,
        }

    def spawn(self, command, name):
        path = self.args.output / name
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        self.children.append(process)
        return process

    def command(self, *parts):
        return [self.args.python, "-m", "modal", *parts]

    def run(self, command, name, *, deadline=None, monitor=False):
        limit = self.deadline if deadline is None else deadline
        require(time.monotonic() < limit)
        process = self.spawn(command, name)
        while process.poll() is None:
            if time.monotonic() >= limit:
                raise TimeoutError("supervised deadline reached")
            if monitor:
                self.check_log()
            time.sleep(0.2)
        if monitor:
            self.check_log()
        require(process.returncode == 0)
        return (self.args.output / name).read_bytes()

    def inventory(self, name, *, deadline=None):
        return inventory_rows(
            self.run(
                self.command("app", "list", "--json"),
                name,
                deadline=deadline,
            )
        )

    def resolve_app(self, rows):
        candidates = [
            row
            for row in rows
            if row["description"] == APP
            and row["app_id"] not in self.previous_ids
            and row["state"] != "stopped"
        ]
        require(len(candidates) == 1)
        require(re.fullmatch(r"ap-[A-Za-z0-9]+", candidates[0]["app_id"]))
        self.app_id = candidates[0]["app_id"]
        self.record["app_id"] = self.app_id
        require(int(candidates[0]["tasks"]) <= 1)

    def check_log(self):
        require(self.log_process is not None and self.log_process.poll() is None)
        path = self.args.output / "platform.log"
        require(path.stat().st_size <= 4_000_000)
        content = path.read_text(errors="replace")
        if re.search(
            r"runner failed|initialization allowance exhausted|outofmemoryerror|"
            r"CUDA out of memory|simple-scene render failed|Traceback",
            content,
            re.I,
        ):
            raise RuntimeError("platform failure observed")
        starts = 0
        for line in content.splitlines():
            offset = line.find('{"simple_scene_stage"')
            if offset < 0:
                continue
            try:
                event = json.loads(line[offset:])["simple_scene_stage"]
            except (ValueError, KeyError):
                continue
            starts += event.get("phase") == "initialization" and event.get("state") == "start"
            if event.get("state") == "failed":
                raise RuntimeError("render failure observed")
        require(starts <= 1)

    def work(self):
        manifest = preflight(self.args)
        rows = self.inventory("apps-before.json")
        require(not any(row["description"] == APP and row["state"] != "stopped" for row in rows))
        self.previous_ids = {row["app_id"] for row in rows}
        write(
            self.args.authorization.parent / "supervisor-attempt.json",
            {
                "authorization_sha256": self.args.authorization_sha256,
                "manifest_sha256": harness.legacy.file_hash(self.args.manifest),
            },
        )
        # Bind dispatch to the same source and ledger after tokenizer/inventory work.
        require(harness.read_manifest(self.args.manifest) == manifest)
        authorization = harness.read_authorization(
            self.args.authorization,
            self.args.authorization_sha256,
            self.args.manifest,
        )
        require(harness.legacy.file_hash(self.args.ledger) == authorization["ledger_sha256"])
        self.deployment_started = True
        self.run(self.command("deploy", str(DEPLOYMENT)), "deploy.log")
        self.resolve_app(self.inventory("apps-deployed.json"))
        self.run(
            [
                self.args.python,
                str(Path(__file__).resolve()),
                "--metadata",
                self.app_id,
                str(self.args.output / "deployment-check.json"),
            ],
            "metadata.log",
        )
        self.log_process = self.spawn(
            self.command(
                "app",
                "logs",
                self.app_id,
                "--follow",
                "--timestamps",
                "--show-container-id",
            ),
            "platform.log",
        )
        ready_at = time.monotonic() + 1
        while time.monotonic() < ready_at:
            self.check_log()
            time.sleep(0.2)
        deadline_unix = min(manifest["expires_at"], time.time() + self.deadline - time.monotonic())
        self.run(
            [
                self.args.python,
                str(Path(harness.__file__).resolve()),
                "--run",
                "--manifest",
                str(self.args.manifest),
                "--authorization",
                str(self.args.authorization),
                "--authorization-sha256",
                self.args.authorization_sha256,
                "--output",
                str(self.args.output / "results"),
                "--deadline-unix",
                str(deadline_unix),
            ],
            "client.log",
            monitor=True,
        )

    def recover_app(self, limit):
        # Allow late deployment visibility, reserving at least 15 seconds to stop it.
        attempt = 0
        discovery_deadline = limit - 16  # One second to reap a read, then 15 for stop.
        while time.monotonic() < discovery_deadline:
            child_offset = len(self.children)
            try:
                rows = self.inventory(
                    f"apps-recovery-{attempt}.json",
                    deadline=min(discovery_deadline, time.monotonic() + 5),
                )
            except (OSError, ValueError, TimeoutError):
                for process in self.children[child_offset:]:
                    if process.poll() is None:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=1)
                self.record["recovery_read_failures"] = (
                    self.record.get("recovery_read_failures", 0) + 1
                )
                rows = []
            candidates = [
                row
                for row in rows
                if row["description"] == APP and row["app_id"] not in self.previous_ids
            ]
            if candidates:
                require(len(candidates) == 1)
                require(re.fullmatch(r"ap-[A-Za-z0-9]+", candidates[0]["app_id"]))
                self.app_id = candidates[0]["app_id"]
                self.record["app_id"] = self.app_id
                return
            attempt += 1
            time.sleep(min(0.5, max(0, discovery_deadline - time.monotonic())))
        raise RuntimeError("new deployment identity unresolved")

    def cleanup(self):
        limit = min(
            self.started + WORK_SECONDS + CLEANUP_SECONDS, time.monotonic() + CLEANUP_SECONDS
        )
        try:
            if self.deployment_started:
                if self.app_id is None:
                    self.recover_app(limit)
                self.record["stop_started_wall_seconds"] = time.monotonic() - self.started
                self.run(
                    self.command("app", "stop", "--yes", self.app_id), "stop.log", deadline=limit
                )
                self.record["stop_returncode"] = 0
                attempt = 0
                while time.monotonic() < limit:
                    rows = self.inventory(f"apps-final-{attempt}.json", deadline=limit)
                    containers = inventory_rows(
                        self.run(
                            self.command("container", "list", "--json"),
                            f"containers-final-{attempt}.json",
                            deadline=limit,
                        )
                    )
                    exact = [row for row in rows if row["app_id"] == self.app_id]
                    require(len(exact) == 1)
                    active = [row for row in containers if row.get("app_id") == self.app_id]
                    require(all("app_id" in row for row in containers))
                    if (
                        exact[0]["state"] == "stopped"
                        and int(exact[0]["tasks"]) == 0
                        and not active
                    ):
                        self.record["external_app_shutdown_verified"] = True
                        break
                    attempt += 1
                    time.sleep(min(1, max(0, limit - time.monotonic())))
                require(self.record["external_app_shutdown_verified"])
        finally:
            for process in self.children:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
            self.record["total_wall_seconds"] = time.monotonic() - self.started


def interrupted(*_):
    raise InterruptedError("supervisor interrupted")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--metadata"]:
        require(len(argv) == 3)
        write(Path(argv[2]), asyncio.run(metadata(argv[1])))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    require(args.output.absolute() == args.output.resolve() and not args.output.exists())
    args.output.mkdir(mode=0o700)
    supervisor = Supervisor(args)
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        supervisor.work()
        supervisor.record["returncode"] = 0
    except BaseException as error:
        supervisor.record.update(returncode=1, failure_type=type(error).__name__)
    finally:
        supervisor.record["work_wall_seconds"] = time.monotonic() - supervisor.started
        try:
            supervisor.cleanup()
        except BaseException as error:
            supervisor.record.update(returncode=1, cleanup_failure_type=type(error).__name__)
        signal.signal(signal.SIGTERM, previous)
        write(args.output / "supervisor.json", supervisor.record)
    print(json.dumps(supervisor.record), flush=True)
    return supervisor.record["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
