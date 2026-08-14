"""E-LiteVPR with the SALAD aggregator, for the Event-VPR benchmark.

Sibling of `elitevpr.py`, which is NOT edited. Everything about the input path
is identical -- the benchmark feeds ImageNet-normalised RGB decoded from frames
written by the `eliteHistogram` reconstruction, which encode our 3-channel event
histogram as R=pos, G=neg, B=(net+1)/2; this wrapper undoes the ImageNet
normalisation, rebuilds the (pos, neg, net) histogram the student was trained
on, and L2-normalises the global descriptor. The student has its own learned
input BatchNorm, so it must receive the raw histogram channels.

Two differences from `elitevpr.py`:

  * the student is `model_salad.EventViTStudentSALAD`, so the descriptor is
    SALAD's optimal-transport aggregation rather than GeM. `forward` returns the
    same `(projected_patches, global_descriptor)` 2-tuple, so nothing else in
    the wrapper changes.
  * there is no pooling to infer -- SALAD has no pooling variant. `elitevpr.py`
    reads clamp/signed/mean off the state-dict keys; here the shape of the
    aggregator is what has to match, and it comes from the env vars below.

SALAD GEOMETRY MUST MATCH TRAINING
----------------------------------
The head's shape is not recorded in a way the checkpoint can be interrogated
for, so it is read from the environment, exactly as `train_megaloc_salad.py`
does when training:

    SALAD_CLUSTERS      (default 64)
    SALAD_CLUSTER_DIM   (default 128)
    SALAD_TOKEN_DIM     (default 256)
    SALAD_SINKHORN      (default 3)

    descriptor dim = SALAD_CLUSTERS * SALAD_CLUSTER_DIM + SALAD_TOKEN_DIM
        default                       64 * 128 + 256 = 8448
        the ckpt_salad1280_* runs     64 *  16 + 256 = 1280   -> SALAD_CLUSTER_DIM=16

A mismatch fails the strict `load_state_dict` below rather than silently
producing garbage, but set it deliberately -- and keep it consistent with
`--descriptors_dimension`, which `parse.py` uses to size retrieval.

WEIGHTS
-------
`ELITEVPR_SALAD_WEIGHTS`, deliberately a different variable from
`ELITEVPR_WEIGHTS`, so a SALAD run cannot pick up a GeM checkpoint (or the other
way round) when both are exported in the same shell.

USAGE
-----
    export ELITEVPR_SALAD_WEIGHTS=/path/to/ckpt_salad1280_1/last_phase1_histogram.pth
    SALAD_CLUSTER_DIM=16 python testing.py --method elitevpr_salad \\
        --reconstruct_method_name eliteHistogram --time_res 1.0 \\
        --seq_len 10 --ref_seq_idx 3 --qry_seq_idx 0

Cached similarity matrices carry the method name, so `elitevpr` and
`elitevpr_salad` never collide. Reconstructions are model-independent and are
shared with every other method.
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

from model_salad import EventViTStudentSALAD

DEFAULT_WEIGHTS = os.environ.get(
    "ELITEVPR_SALAD_WEIGHTS",
    os.path.join(_BENCH, "elitevpr_weights", "best_phase1_salad.pth"))

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_SALAD_ENV = {
    "num_clusters": ("SALAD_CLUSTERS", 64),
    "cluster_dim": ("SALAD_CLUSTER_DIM", 128),
    "token_dim": ("SALAD_TOKEN_DIM", 256),
    "sinkhorn_iters": ("SALAD_SINKHORN", 3),
}


def _salad_params():
    return {k: int(os.environ.get(env, default))
            for k, (env, default) in _SALAD_ENV.items()}


class ElitevprSaladModel(nn.Module):
    def __init__(self, weights_path=DEFAULT_WEIGHTS, img_size=(384, 384)):
        super().__init__()
        print(f"[elitevpr_salad] loading weights: {weights_path}", flush=True)
        state = torch.load(weights_path, map_location="cpu")
        # unwrap training checkpoints saved as {"model_state[_dict]": ...}
        if isinstance(state, dict):
            for k in ("model_state_dict", "model_state", "state_dict"):
                if k in state:
                    state = state[k]
                    break
        # Phase-2 style wrappers keep the student under a "student." prefix.
        if any(k.startswith("student.") for k in state):
            state = {k[len("student."):]: v for k, v in state.items()
                     if k.startswith("student.")}

        p = _salad_params()
        dim = p["num_clusters"] * p["cluster_dim"] + p["token_dim"]
        print(f"[elitevpr_salad] SALAD {p['num_clusters']} clusters x "
              f"{p['cluster_dim']} + {p['token_dim']} token = {dim}-d "
              f"descriptor (sinkhorn {p['sinkhorn_iters']})", flush=True)
        if not any(k.startswith("salad.") for k in state):
            raise RuntimeError(
                "checkpoint has no salad.* keys -- this looks like a GeM "
                "student. Use --method elitevpr with ELITEVPR_WEIGHTS instead.")

        self.student = EventViTStudentSALAD(
            teacher_dim=1024, num_patches=576,
            img_size=tuple(img_size), in_channels=3, **p)
        self.student.load_state_dict(state)
        self.student.eval()
        self.desc_dim = dim
        self.img_size = tuple(img_size)
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        # x: ImageNet-normalised RGB (B, 3, H, W) from TestDataset
        if x.shape[-2:] != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode="bilinear",
                              align_corners=False)
        rgb = x * self.std + self.mean            # -> [0, 1] uint8-equivalent
        pos = rgb[:, 0:1]
        neg = rgb[:, 1:2]
        net = rgb[:, 2:3] * 2.0 - 1.0
        hist = torch.cat([pos, neg, net], dim=1)  # our 3-channel histogram
        _, g = self.student(hist)
        return F.normalize(g, p=2, dim=-1)


def get_model(weights_path=DEFAULT_WEIGHTS, img_size=(384, 384)):
    return ElitevprSaladModel(weights_path, img_size)
