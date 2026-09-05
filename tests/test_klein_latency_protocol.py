import json
import struct

import pytest

from deploy.klein_latency_protocol import MAX_RESPONSE_BYTES, pack_response, unpack_response


def test_wire_roundtrip_rejects_corruption_and_nonstandard_metadata():
    payload = {
        "request_id": "a" * 32,
        "metrics": {"seconds": 0.25},
        "master": b"jpeg",
        "depth": b"depth",
    }
    encoded = pack_response(payload)
    assert unpack_response(encoded) == payload
    malformed_headers = (
        b'{"asset_lengths":[0,0],"seconds":NaN}',
        b'{"asset_lengths":[0,0],"asset_lengths":[0,0]}',
        json.dumps({"asset_lengths": [True, 0]}).encode(),
    )
    malformed = [encoded[:-1], encoded + b"x", b"x" * (MAX_RESPONSE_BYTES + 1)]
    malformed.extend(b"BFL1" + struct.pack(">I", len(h)) + h for h in malformed_headers)
    for raw in malformed:
        with pytest.raises(ValueError):
            unpack_response(raw)
    with pytest.raises(ValueError):
        pack_response({"master": b"x" * MAX_RESPONSE_BYTES})
