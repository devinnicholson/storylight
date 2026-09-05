"""Use the existing bounded client against the two region-controlled deployments."""

import asyncio
import importlib
from urllib.parse import urlsplit

from bookforge.klein_latency_client import LatencyClient, LatencyError


class RegionClient(LatencyClient):
    def __init__(self, expected: dict, **kwargs):
        super().__init__(
            {**expected, "deployment_sha256": expected["region_deployment_sha256"]}, **kwargs
        )

    async def lookup(self, transport: str) -> None:
        if self.modal is None:
            self.modal = importlib.import_module("modal")
        if transport == "sdk" and self.instance is None:
            cls = self.modal.Cls.from_name("bookforge-klein-region-sdk", "RegionStudio")
            await asyncio.wait_for(cls.hydrate.aio(), 10)
            self.instance = cls()
        elif transport == "http" and self.url is None:
            server = self.modal.Server.from_name("bookforge-klein-region-http", "RegionServer")
            url = await asyncio.wait_for(server.get_url.aio(), 10)
            parsed = urlsplit(url or "")
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or not parsed.hostname.endswith(".us-east.modal.direct")
                or parsed.username is not None
                or parsed.password is not None
                or parsed.port
                or parsed.query
                or parsed.fragment
                or parsed.path not in {"", "/"}
            ):
                raise LatencyError("region_server_url_invalid")
            self.url = url.rstrip("/")

    async def invoke(self, transport: str, request: dict, expected_bucket: int | None):
        payload, timings = await super().invoke(transport, request, expected_bucket)
        location = payload["location"]
        if (
            location["cloud"] != "CLOUD_PROVIDER_AWS"
            or location["compute_region"] != "us-east-1"
            or location["routing_region"] != "us-east"
        ):
            self.last_failure = "region_response_provenance_mismatch"
            raise LatencyError(self.last_failure)
        return payload, timings
