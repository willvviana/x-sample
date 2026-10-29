"""Export the Beat This! small0 checkpoint (MIT) to the two ONNX files the engine uses.

Not part of the engine environment: torch and beat_this are needed only here. Run it in an
isolated, temporary virtual environment with the pinned versions below, on a checkpoint
that was downloaded with approval and matches CHECKPOINT_SHA256:

    uv venv --python 3.12 <tmp>/.venv
    uv pip install --python <tmp>/.venv/Scripts/python.exe \
        --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
    uv pip install --python <tmp>/.venv/Scripts/python.exe einops==0.8.2 \
        rotary-embedding-torch==0.9.1 soxr==1.1.0 onnx==1.23.2 onnxscript==0.7.2 \
        numpy==2.5.4 torchaudio==2.11.1 --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        "beat_this @ git+https://github.com/CPJKU/beat_this@b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c"
    <tmp>/.venv/Scripts/python.exe -I -X utf8 tools/export_beat_this.py small0.ckpt <out_dir>

Then install the files with `uv run python tools/install_beat_this.py <out_dir>`.

Two deviations from the upstream module are needed for torch.export, both checked here to
give identical outputs: SumHead's torch.autocast block is replaced by the plain float32 sum,
and rotary_embedding_torch's rotation (autocast decorators, in-place frequency cache) is
replaced by the same arithmetic without them.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import rotary_embedding_torch as ret
import torch
from beat_this.inference import load_model

CHECKPOINT_SHA256 = "6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b"
CHUNK, N_MELS, OPSET = 1500, 128, 18


class Logits(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.m = model

    def forward(self, spect: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.m.transformer_blocks(self.m.frontend(spect))
        bd = self.m.task_heads.beat_downbeat_lin(h)
        downbeat = bd[..., 1]
        return bd[..., 0] + downbeat, downbeat


def rotate(self, t, seq_dim=None, offset=0, scale=None):  # type: ignore[no-untyped-def]
    # rotary_embedding_torch 0.9.1 with rot_dim == head_dim, no xpos, offset 0.
    n = t.shape[-2]
    pos = torch.arange(n, dtype=t.dtype)
    freqs = torch.repeat_interleave(pos[:, None] * self.freqs[None, :], 2, dim=-1)
    pairs = t.unflatten(-1, (-1, 2))
    rot = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    return t * freqs.cos() + rot * freqs.sin()


def main() -> None:
    ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint SHA-256 mismatch: {digest}")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = load_model(str(ckpt), "cpu")
    reference = model.eval()
    probe = torch.rand(1, CHUNK, N_MELS) * 4
    with torch.no_grad():
        expected = reference(probe)
        ret.RotaryEmbedding.rotate_queries_or_keys = rotate
        wrapped = Logits(model).eval().requires_grad_(False)
        beat, down = wrapped(probe)
        for a, b in ((beat, expected["beat"]), (down, expected["downbeat"])):
            if not torch.allclose(a, b, atol=1e-5, rtol=0):
                raise SystemExit("patched model differs from upstream")
        time_dim = torch.export.Dim("time", min=8, max=CHUNK)
        for name, shapes in (
            ("beat_this_small0_t1500.onnx", None),
            ("beat_this_small0.onnx", {"spect": {1: time_dim}}),
        ):
            torch.onnx.export(
                wrapped,
                (probe,),
                str(out / name),
                input_names=["spect"],
                output_names=["beat", "downbeat"],
                dynamic_shapes=shapes,
                opset_version=OPSET,
                dynamo=True,
                external_data=False,
            )
        np.savez(
            out / "export_check.npz",
            spect=probe.numpy(),
            beat=expected["beat"].numpy(),
            downbeat=expected["downbeat"].numpy(),
        )
    for name in ("beat_this_small0_t1500.onnx", "beat_this_small0.onnx"):
        data = (out / name).read_bytes()
        print(name, len(data), hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
