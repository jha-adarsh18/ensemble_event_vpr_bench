"""E-LiteVPR feature extractor for the Event-VPR benchmark.

The benchmark feeds every model ImageNet-normalised RGB tensors decoded from
the stored frames. E-LiteVPR's frames (reconstruction method `eliteHistogram`)
encode our 3-channel event histogram as R=pos, G=neg, B=(net+1)/2. This
wrapper undoes the ImageNet normalisation, reconstructs the (pos, neg, net)
histogram the ViT student was trained on, runs the student, and returns the
L2-normalised GeM global descriptor (dim = teacher_dim = 1024).

The student has its own learned input BatchNorm, so it must receive the raw
histogram channels -- not ImageNet-normalised values.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# elitevpr.py lives at
# <repo>/external/ensemble_event_vpr_bench/vpr_methods_evaluation/vpr_models/
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     "..", "..", "..", ".."))
_BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from model import EventViTStudent

DEFAULT_WEIGHTS = os.path.join(_BENCH, "elitevpr_weights",
                               "merged_alpha0.5.pth")

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class ElitevprModel(nn.Module):
    def __init__(self, weights_path=DEFAULT_WEIGHTS, img_size=(384, 384)):
        super().__init__()
        self.student = EventViTStudent(
            teacher_dim=1024, num_patches=576,
            img_size=tuple(img_size), in_channels=3)
        state = torch.load(weights_path, map_location="cpu")
        self.student.load_state_dict(state)
        self.student.eval()
        self.img_size = tuple(img_size)
        self.register_buffer("mean",
                             torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",
                             torch.tensor(_STD).view(1, 3, 1, 1))

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
    return ElitevprModel(weights_path, img_size)
