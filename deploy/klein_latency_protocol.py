"""Bounded wire format for the isolated renderer transport comparison."""

from __future__ import annotations

import hashlib
import json
import re
import struct

MAX_REQUEST_BYTES = 8192
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_METADATA_BYTES = 65536


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_request(request: dict) -> None:
    if not isinstance(request, dict) or set(request) != {
        "request_id",
        "operation",
        "prompt",
        "seed",
    }:
        raise ValueError("invalid request fields")
    if not isinstance(request["request_id"], str) or not re.fullmatch(
        r"[a-f0-9]{32}", request["request_id"]
    ):
        raise ValueError("invalid request identity")
    if type(request["seed"]) is not int or not 0 <= request["seed"] <= 2**32 - 1:
        raise ValueError("invalid seed")
    if not isinstance(request["prompt"], str):
        raise ValueError("invalid prompt type")
    if request["operation"] == "prewarm":
        if request["prompt"] != "" or request["seed"] != 0:
            raise ValueError("invalid warmup")
    elif (
        request["operation"] != "render"
        or not request["prompt"].strip()
        or len(request["prompt"]) > 4000
    ):
        raise ValueError("invalid operation")


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def decode_json(raw: bytes) -> dict:
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _invalid_constant(_: str):
    raise ValueError("non-finite JSON constant")


def pack_response(payload: dict) -> bytes:
    metadata = {key: value for key, value in payload.items() if key not in ("master", "depth")}
    master, depth = payload.get("master", b""), payload.get("depth", b"")
    if not isinstance(master, bytes) or not isinstance(depth, bytes):
        raise ValueError("invalid asset bytes")
    metadata["asset_lengths"] = [len(master), len(depth)]
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if (
        len(header) > MAX_METADATA_BYTES
        or 8 + len(header) + len(master) + len(depth) > MAX_RESPONSE_BYTES
    ):
        raise ValueError("response exceeds bound")
    return b"BFL1" + struct.pack(">I", len(header)) + header + master + depth


def unpack_response(raw: bytes) -> dict:
    if not isinstance(raw, bytes) or not 8 <= len(raw) <= MAX_RESPONSE_BYTES or raw[:4] != b"BFL1":
        raise ValueError("invalid response frame")
    size = struct.unpack(">I", raw[4:8])[0]
    if not 0 < size <= MAX_METADATA_BYTES or 8 + size > len(raw):
        raise ValueError("invalid metadata length")
    payload = decode_json(raw[8 : 8 + size])
    lengths = payload.pop("asset_lengths", None)
    if (
        not isinstance(lengths, list)
        or len(lengths) != 2
        or any(type(n) is not int or n < 0 for n in lengths)
    ):
        raise ValueError("invalid asset lengths")
    if 8 + size + sum(lengths) != len(raw) or "master" in payload or "depth" in payload:
        raise ValueError("invalid response length")
    boundary = 8 + size + lengths[0]
    payload.update(master=raw[8 + size : boundary], depth=raw[boundary:])
    return payload
