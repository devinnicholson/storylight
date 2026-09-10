from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path

TCP_ESTABLISHED = "01"
TCP_SYN_SENT = "02"
TCP_SYN_RECV = "03"
TCP_LISTEN = "0A"


def audit_process(pid: int, proc_root: Path = Path("/proc")) -> dict[str, object]:
    process_root = proc_root / str(pid)
    if not process_root.exists():
        return {"ready": False, "pid": pid, "detail": "process is not visible", "connections": []}

    socket_inodes, fd_error = _socket_inodes(process_root / "fd")
    connections: list[dict[str, object]] = []
    visibility_errors = [fd_error] if fd_error else []
    for filename, family, protocol in (
        ("tcp", 4, "tcp"),
        ("tcp6", 6, "tcp"),
        ("udp", 4, "udp"),
        ("udp6", 6, "udp"),
    ):
        observed, error = _read_connections(
            process_root / "net" / filename, family, protocol, socket_inodes
        )
        connections.extend(observed)
        if error:
            visibility_errors.append(error)

    unsafe = [connection for connection in connections if connection["unsafe"]]
    ready = not visibility_errors and not unsafe
    return {
        "ready": ready,
        "pid": pid,
        "detail": (
            "; ".join(visibility_errors)
            if visibility_errors
            else "no non-loopback listeners or active TCP/UDP connections"
            if not unsafe
            else f"{len(unsafe)} unsafe non-loopback socket(s)"
        ),
        "connections": connections,
        "visibility_errors": visibility_errors,
    }


def _socket_inodes(fd_root: Path) -> tuple[set[str], str | None]:
    inodes: set[str] = set()
    try:
        descriptors = list(fd_root.iterdir())
    except OSError as error:
        return inodes, f"cannot inspect {fd_root}: {error}"
    for descriptor in descriptors:
        try:
            target = os.readlink(descriptor)
        except FileNotFoundError:
            # File descriptors can legitimately close between iterdir() and
            # readlink(). Other read failures mean socket visibility is not
            # trustworthy and must fail the privacy audit closed.
            continue
        except OSError as error:
            return inodes, f"cannot inspect {descriptor}: {error}"
        if target.startswith("socket:[") and target.endswith("]"):
            inodes.add(target[8:-1])
    return inodes, None


def _read_connections(
    path: Path, family: int, protocol: str, inodes: set[str]
) -> tuple[list[dict[str, object]], str | None]:
    try:
        rows = path.read_text(encoding="ascii").splitlines()[1:]
    except OSError as error:
        return [], f"cannot inspect {path}: {error}"

    connections: list[dict[str, object]] = []
    for row in rows:
        fields = row.split()
        if len(fields) < 10 or fields[9] not in inodes:
            continue
        local_host, local_port = _decode_address(fields[1], family)
        remote_host, remote_port = _decode_address(fields[2], family)
        state = fields[3]
        local_address = ipaddress.ip_address(local_host)
        remote_address = ipaddress.ip_address(remote_host)
        if protocol == "tcp" and state == TCP_LISTEN:
            unsafe = not local_address.is_loopback
        elif protocol == "tcp" and state in {TCP_ESTABLISHED, TCP_SYN_SENT, TCP_SYN_RECV}:
            unsafe = not remote_address.is_loopback
        elif protocol == "udp":
            unsafe = (not local_address.is_loopback) or (
                not remote_address.is_unspecified and not remote_address.is_loopback
            )
        else:
            unsafe = False
        connections.append(
            {
                "protocol": protocol,
                "family": family,
                "state": state,
                "local_host": local_host,
                "local_port": local_port,
                "remote_host": remote_host,
                "remote_port": remote_port,
                "remote_loopback": remote_address.is_loopback,
                "unsafe": unsafe,
            }
        )
    return connections, None


def _decode_address(encoded: str, family: int) -> tuple[str, int]:
    host_hex, port_hex = encoded.split(":", 1)
    raw = bytes.fromhex(host_hex)
    if family == 4:
        host = str(ipaddress.IPv4Address(raw[::-1]))
    else:
        words = [raw[index : index + 4][::-1] for index in range(0, 16, 4)]
        host = str(ipaddress.IPv6Address(b"".join(words)))
    return host, int(port_hex, 16)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Storylight process TCP privacy boundary")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = audit_process(arguments.pid)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(f"{payload}\n", encoding="utf-8")
    print(payload)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
