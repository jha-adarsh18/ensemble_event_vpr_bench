"""E-LiteVPR event representation as an Event-VPR-bench reconstruction method.

Produces E-LiteVPR's own 3-channel event histogram (pos unit-max, neg
unit-max, net polarity in [-1, 1]) -- byte-for-byte the same as
scripts/brisbane_representation.py::process_event_histogram -- and stores it
as a uint8 RGB frame:

    R = pos * 255,   G = neg * 255,   B = (net + 1) / 2 * 255

The `elitevpr` vpr_model de-quantises these channels back to (pos, neg, net)
before running the ViT student. Frames are saved as PNG (lossless) by
load_and_save.py for this method, so the signed net channel is preserved.
"""
import os
import sys

import numpy as np
from tqdm import tqdm

from reconstruction.base_reconstructor import BaseReconstructor

# Make the E-LiteVPR repo importable: this file lives at
# <repo>/external/ensemble_event_vpr_bench/reconstruction/eliteHistogram.py
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from brisbane_representation import (compute_hot_pixel_mask,
                                     process_event_histogram)

OUT_SIZE = (384, 384)   # (W, H) passed to cv2.resize inside the histogram


class EventReconstructor(BaseReconstructor):
    def __init__(self):
        pass

    def reconstruct(self, eventsData, sensor_size, start_indices, end_indices,
                    hp_loc=None, hot_mask=None):
        """
        Args:
            eventsData: structured array with fields 'x', 'y', 'p' (p in {0,1}).
            sensor_size: (width, height).
            start_indices/end_indices: per-frame event window bounds.
            hot_mask: optional precomputed per-traverse hot-pixel mask. Supplied
                by the streaming NSAVP loader (which computes it over the full
                traverse in a chunked pre-pass); when None it is computed here
                over eventsData, as in the standard full-load path.
        Returns:
            frames: (num_frames, 384, 384, 3) uint8 RGB.
            frame_times: list of (start_idx, end_idx).
        """
        width, height = int(sensor_size[0]), int(sensor_size[1])
        sensor_hw = (height, width)

        # Hot-pixel mask over the whole traverse (as in the E-LiteVPR pipeline)
        if hot_mask is None:
            x_all = eventsData['x'].astype(np.int64)
            y_all = eventsData['y'].astype(np.int64)
            hot_mask = compute_hot_pixel_mask(x_all, y_all, sensor_hw,
                                              threshold=99.5)

        frames = np.zeros((len(start_indices), OUT_SIZE[1], OUT_SIZE[0], 3),
                          dtype=np.uint8)
        frame_times = []
        for i, (s, e) in enumerate(tqdm(zip(start_indices, end_indices),
                                        total=len(start_indices),
                                        desc="E-LiteVPR histogram")):
            frame_times.append((s, e))
            x = eventsData['x'][s:e].astype(np.int64)
            y = eventsData['y'][s:e].astype(np.int64)
            p = eventsData['p'][s:e].astype(np.int64)

            hist = process_event_histogram(x, y, p, hot_mask, OUT_SIZE,
                                           sensor_hw)          # (3,H,W) float
            pos, neg, net = hist[0], hist[1], hist[2]
            rgb = frames[i]
            rgb[:, :, 0] = np.clip(pos * 255.0, 0, 255).astype(np.uint8)
            rgb[:, :, 1] = np.clip(neg * 255.0, 0, 255).astype(np.uint8)
            rgb[:, :, 2] = np.clip((net + 1.0) * 0.5 * 255.0, 0, 255
                                   ).astype(np.uint8)
        return frames, frame_times
