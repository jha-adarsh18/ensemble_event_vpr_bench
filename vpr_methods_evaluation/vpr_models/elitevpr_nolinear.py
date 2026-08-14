"""E-LiteVPR WITHOUT the patch projection -- the `no_linear_proj` ablation arm.

Sibling of `elitevpr.py`, which is NOT edited, and neither is
`scripts/model.py`. The input path is identical: the benchmark feeds
ImageNet-normalised RGB decoded from `eliteHistogram` frames (R=pos, G=neg,
B=(net+1)/2); this undoes that, rebuilds the (pos, neg, net) histogram, runs the
student and L2-normalises the descriptor.

HOW THE ARCHITECTURE IS CHANGED WITHOUT A NEW MODEL FILE
--------------------------------------------------------
`EventViTStudent.forward` does `self.proj(patches)` and then pools. The
ablation removes that Linear, so the student pools the raw backbone patches and
the descriptor is `student_dim` (384 for ViT-S) rather than `teacher_dim`
(1024). Rather than mutate `model.py`, this builds the canonical student and
replaces `proj` with `nn.Identity()`:

  * `nn.Identity` has NO parameters, so the model's state_dict contains no
    `proj.*` keys and a strict `load_state_dict` of a proj-less checkpoint
    succeeds -- while a NORMAL checkpoint still fails loudly on unexpected
    `proj.weight` / `proj.bias`, which is the desired guard.
  * pooling is width-agnostic: both `GeM.p` and `SignedGeM.p` are
    `nn.Parameter(torch.ones(1))` scalars (`model.py:16`, `:45`), so nothing
    downstream assumes 1024.

DESCRIPTOR WIDTH
----------------
384, and `parse.py` must agree -- it sizes the retrieval array, and a mismatch
surfaces as
    ValueError: shape mismatch: value array of shape (N,384) could not be
    broadcast to indexing result of shape (N,1024)
`ELITEVPR_NOLINEAR_DIM` overrides it if a different backbone is ever used.

WEIGHTS
-------
`ELITEVPR_NOLINEAR_WEIGHTS`, a separate variable from `ELITEVPR_WEIGHTS`, so
this arm cannot pick up a normal checkpoint when both are exported.

USAGE
-----
    export ELITEVPR_NOLINEAR_WEIGHTS=/workspace/E-LiteVPR/model_weights/\\
elitevpr_no_linear_proj_3ds_s42_e30.pth
    python testing.py --method elitevpr_nolinear \\
        --reconstruct_method_name eliteHistogram --time_res 1.0 \\
        --seq_len 10 --ref_seq_idx 6 --qry_seq_idx 7

Cached similarity matrices carry the method name, so this never collides with
`elitevpr` or `elitevpr_salad`. Reconstructions are shared.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# this file lives at
# <repo>/external/ensemble_event_vpr_bench/vpr_methods_evaluation/vpr_models/
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     "..", "..", "..", ".."))
_BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from model import EventViTStudent

DEFAULT_WEIGHTS = os.environ.get(
    "ELITEVPR_NOLINEAR_WEIGHTS",
    os.path.join(_BENCH, "elitevpr_weights", "best_phase1_nolinear.pth"))
DESC_DIM = int(os.environ.get("ELITEVPR_NOLINEAR_DIM", 384))

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class ElitevprNoLinearModel(nn.Module):
    def __init__(self, weights_path=DEFAULT_WEIGHTS, img_size=(384, 384)):
        super().__init__()
        print(f"[elitevpr_nolinear] loading weights: {weights_path}", flush=True)
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict):
            for k in ("model_state_dict", "model_state", "state_dict"):
                if k in state:
                    state = state[k]
                    break
        if any(k.startswith("student.") for k in state):
            state = {k[len("student."):]: v for k, v in state.items()
                     if k.startswith("student.")}

        if any(k.startswith("proj.") for k in state):
            raise RuntimeError(
                "checkpoint HAS proj.* keys -- this is a normal student. Use "
                "--method elitevpr with ELITEVPR_WEIGHTS instead.")

        # same pooling inference as elitevpr.py: the state-dict key encodes it
        if any(k.startswith("gem.") for k in state):
            pooling = "clamp"
        elif any(k.startswith("pool.") for k in state):
            pooling = "signed"
        else:
            pooling = "mean"
        print(f"[elitevpr_nolinear] pooling inferred from checkpoint: {pooling}",
              flush=True)

        self.student = EventViTStudent(
            teacher_dim=1024, num_patches=576,
            img_size=tuple(img_size), in_channels=3, pooling=pooling)
        # the ablation: pool the raw backbone patches, no projection.
        # Identity has no parameters, so no proj.* keys are expected.
        self.student.proj = nn.Identity()
        self.student.load_state_dict(state)
        self.student.eval()
        self.img_size = tuple(img_size)
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))
        print(f"[elitevpr_nolinear] descriptor dim {DESC_DIM} "
              f"(no projection; pooling over backbone patches)", flush=True)

    @torch.no_grad()
    def forward(self, x):
        if x.shape[-2:] != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode="bilinear",
                              align_corners=False)
        rgb = x * self.std + self.mean
        pos = rgb[:, 0:1]
        neg = rgb[:, 1:2]
        net = rgb[:, 2:3] * 2.0 - 1.0
        hist = torch.cat([pos, neg, net], dim=1)
        _, g = self.student(hist)
        return F.normalize(g, p=2, dim=-1)


def get_model(weights_path=DEFAULT_WEIGHTS, img_size=(384, 384)):
    return ElitevprNoLinearModel(weights_path, img_size)
