"""Exclusive-use CUDA Graph comparison around already warmed regional Klein blocks.

Only the two frozen text-only shapes are admitted. Capture failures are terminal;
measurements never capture, compile, or fall back to ordinary execution.
"""

import hashlib
import json
import time
from contextlib import contextmanager

TENSORS = ("hidden_states", "encoder_hidden_states", "timestep", "img_ids", "txt_ids")


def require(value):
    if not value:
        raise ValueError("denoiser graph requires the qualified static Klein workload")


class DenoiserGraphAdapter:
    def __init__(self, transformer):
        import diffusers
        import torch
        from diffusers import Flux2Transformer2DModel

        require(str(torch.__version__) == "2.8.0+cu128" and diffusers.__version__ == "0.39.0")
        require(type(transformer) is Flux2Transformer2DModel and not transformer.training)
        blocks = [*transformer.transformer_blocks, *transformer.single_transformer_blocks]
        require(
            len(blocks) == 25 and all(block._compiled_call_impl is not None for block in blocks)
        )
        require("forward" not in vars(transformer))
        self.torch, self.transformer = torch, transformer
        self.original = transformer.forward
        self.graphs = {}
        self.stream_id = None
        self.active = self.failed = False
        self.capture_warmup_calls = self.captured_forward_calls = 0

    def _signature(self, args, kwargs):
        torch = self.torch
        require(not args and torch.is_inference_mode_enabled() and not torch.is_grad_enabled())
        require(set(kwargs) == {*TENSORS, "guidance", "joint_attention_kwargs", "return_dict"})
        require(kwargs["guidance"] is None and kwargs["joint_attention_kwargs"] is None)
        require(kwargs["return_dict"] is False)
        require(all(torch.is_tensor(kwargs[name]) for name in TENSORS))
        bucket = kwargs["encoder_hidden_states"].shape[1]
        require(bucket in (128, 256))
        shapes = ((1, 2304, 128), (1, bucket, 7680), (1,), (1, 2304, 4), (1, bucket, 4))
        signature = []
        device = kwargs["hidden_states"].device
        require(device.type == "cuda")
        stream_id = torch.cuda.current_stream(device).cuda_stream
        require(self.stream_id is None or self.stream_id == stream_id)
        self.stream_id = stream_id
        for name, shape in zip(TENSORS, shapes, strict=True):
            tensor = kwargs[name]
            dtype = torch.int64 if name.endswith("ids") else torch.bfloat16
            require(tuple(tensor.shape) == shape and tensor.dtype == dtype)
            require(
                tensor.device == device
                and not tensor.requires_grad
                and tensor.layout == torch.strided
            )
            signature.append((name, shape, tuple(tensor.stride()), str(dtype), str(device)))
        return bucket, tuple(signature)

    def _capture(self, bucket, signature, kwargs):
        require(
            len(self.graphs) < 2
            and not any(row["bucket"] == bucket for row in self.graphs.values())
        )
        torch = self.torch
        started = time.perf_counter()
        static = {name: kwargs[name].clone() for name in TENSORS}
        static.update(guidance=None, joint_attention_kwargs=None, return_dict=False)
        require(self._signature((), static)[1] == signature)
        stream = torch.cuda.Stream(device=static["hidden_states"].device)
        current = torch.cuda.current_stream(static["hidden_states"].device)
        stream.wait_stream(current)
        for name in TENSORS:
            static[name].record_stream(stream)
        with torch.cuda.stream(stream):
            for _ in range(3):
                warm = self.original(**static)
                self.capture_warmup_calls += 1
                del warm
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.compiler.set_stance("fail_on_recompile"), torch.cuda.graph(graph, stream=stream):
            output = self.original(**static)
        self.captured_forward_calls += 1
        require(isinstance(output, tuple) and len(output) == 1 and torch.is_tensor(output[0]))
        require(output[0].shape == static["hidden_states"].shape)
        current.wait_stream(stream)
        stream.synchronize()
        self.graphs[signature] = {
            "bucket": bucket,
            "static": static,
            "output": output,
            "graph": graph,
            "capture_seconds": time.perf_counter() - started,
            "replays": 0,
            "signature_sha256": hashlib.sha256(json.dumps(signature).encode()).hexdigest(),
        }

    def _invoke(self, allow_capture, args, kwargs):
        require(not self.failed)
        try:
            bucket, signature = self._signature(args, kwargs)
            if signature not in self.graphs:
                require(allow_capture)
                self._capture(bucket, signature, kwargs)
            row = self.graphs[signature]
            for name in TENSORS:
                row["static"][name].copy_(kwargs[name])
            row["graph"].replay()
            row["replays"] += 1
            # Graph output storage is overwritten on replay; callers must own their result.
            return (row["output"][0].clone(),)
        except BaseException:
            self.failed = True
            raise

    @contextmanager
    def _using(self, allow_capture):
        require(not self.active and not self.failed and "forward" not in vars(self.transformer))
        if not allow_capture:
            require(len(self.graphs) == 2)
        self.active = True

        def forward(*args, **kwargs):
            return self._invoke(allow_capture, args, kwargs)

        object.__setattr__(self.transformer, "forward", forward)
        try:
            yield self
        finally:
            object.__delattr__(self.transformer, "forward")
            self.active = False

    def capture(self):
        return self._using(True)

    def replay(self):
        return self._using(False)

    def report(self):
        return {
            "graph_count": len(self.graphs),
            "capture_warmup_calls": self.capture_warmup_calls,
            "captured_forward_calls": self.captured_forward_calls,
            "graphs": [
                {
                    key: row[key]
                    for key in ("bucket", "signature_sha256", "capture_seconds", "replays")
                }
                for row in sorted(self.graphs.values(), key=lambda item: item["bucket"])
            ],
        }
