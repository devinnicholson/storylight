from pydantic import BaseModel, ConfigDict


class RuntimeComponent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ready: bool
    name: str
    detail: str


class RuntimeStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ready: bool
    environment: str
    version: str
    loopback_only: bool
    model: RuntimeComponent
    asr: RuntimeComponent
    storage: RuntimeComponent
    data_dir: str
    cache_dir: str
