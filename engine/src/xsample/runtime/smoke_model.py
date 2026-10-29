"""Tiny, locally generated ONNX model used for backend probes and smoke tests.

Graph (stereo audio shaped [1, 2, T]):
    y = tanh(Conv1x1(Relu(Conv9(x)))) + x
No weights are downloaded; they come from a fixed seed so results are reproducible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import onnx
from onnx import TensorProto, helper, numpy_helper

MODEL_ID = "xsample.smoke.conv1d"
OPSET = 17
IR_VERSION = 9
CHANNELS = 2
HIDDEN = 16
KERNEL = 9

F32 = npt.NDArray[np.float32]


@dataclass(frozen=True)
class SmokeModel:
    chunk_samples: int
    model_bytes: bytes
    w1: F32
    b1: F32
    w2: F32
    b2: F32

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.model_bytes).hexdigest()

    def reference(self, x: F32) -> F32:
        """NumPy reference implementation, accumulated in float64."""
        x64 = x[0].astype(np.float64)
        pad = KERNEL // 2
        padded = np.pad(x64, ((0, 0), (pad, pad)))
        t = x64.shape[1]
        hidden = np.zeros((HIDDEN, t), dtype=np.float64)
        w1 = self.w1.astype(np.float64)
        for k in range(KERNEL):
            hidden += w1[:, :, k] @ padded[:, k : k + t]
        hidden += self.b1.astype(np.float64)[:, None]
        np.maximum(hidden, 0.0, out=hidden)
        out = self.w2[:, :, 0].astype(np.float64) @ hidden + self.b2.astype(np.float64)[:, None]
        result: F32 = (np.tanh(out) + x64)[None].astype(np.float32)
        return result


def build(chunk_samples: int, seed: int = 1234) -> SmokeModel:
    if chunk_samples < KERNEL:
        raise ValueError("chunk_samples too small")
    rng = np.random.default_rng(seed)
    w1 = (rng.standard_normal((HIDDEN, CHANNELS, KERNEL)) * 0.2).astype(np.float32)
    b1 = (rng.standard_normal(HIDDEN) * 0.05).astype(np.float32)
    w2 = (rng.standard_normal((CHANNELS, HIDDEN, 1)) * 0.2).astype(np.float32)
    b2 = (rng.standard_normal(CHANNELS) * 0.05).astype(np.float32)

    shape = [1, CHANNELS, chunk_samples]
    nodes = [
        helper.make_node("Conv", ["x", "w1", "b1"], ["h"], pads=[KERNEL // 2, KERNEL // 2]),
        helper.make_node("Relu", ["h"], ["hr"]),
        helper.make_node("Conv", ["hr", "w2", "b2"], ["o"]),
        helper.make_node("Tanh", ["o"], ["ot"]),
        helper.make_node("Add", ["ot", "x"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "xsample_smoke",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
        [
            numpy_helper.from_array(w1, "w1"),
            numpy_helper.from_array(b1, "b1"),
            numpy_helper.from_array(w2, "w2"),
            numpy_helper.from_array(b2, "b2"),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="xsample"
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    return SmokeModel(chunk_samples, model.SerializeToString(), w1, b1, w2, b2)


def build_cpu_only_op_model() -> bytes:
    """A model whose `Det` node has no DirectML kernel; used to test fallback reporting."""
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["r"]), helper.make_node("Det", ["r"], ["y"])],
        "xsample_fallback",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 3, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return bytes(model.SerializeToString())


def test_signal(chunk_index: int, chunk_samples: int, sample_rate: int = 44100) -> F32:
    """Deterministic stereo test chunk: two sines plus low-level noise, in [-1, 1]."""
    start = chunk_index * chunk_samples
    t = (np.arange(start, start + chunk_samples, dtype=np.float64)) / sample_rate
    rng = np.random.default_rng(chunk_index)
    left = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.4 * np.sin(2 * np.pi * 659.25 * t)
    noise = rng.standard_normal((2, chunk_samples)) * 0.01
    return (np.stack([left, right]) + noise)[None].astype(np.float32)
