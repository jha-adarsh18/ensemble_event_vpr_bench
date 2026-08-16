"""Second E-LiteVPR student under its own method name, for MODEL-DIVERSITY
ensembling.

WHY THIS EXISTS
---------------
`ablate_ensembles.py:414` builds each member's similarity-matrix path as

    logs/fixed_timebins_{dt}/{ref}_vs_{qry}_{VPR_METHOD}_l2_reconstruct_{recon}_{dt}_{mode}.npy

so the member's identity in the cache is its METHOD NAME. Two students both
running as `--method elitevpr` write the same filenames and silently overwrite
one another, and `--ensemble_over vpr` iterates over `--vpr_methods`, which are
method names. Ensembling two checkpoints therefore requires a second registered
method, not a second env var alone.

This is that second method. It is the SAME model class as `elitevpr` -- the
MixVPR-distilled student is 1024-d like the MegaLoc-distilled one, so
`elitevpr.py`'s construction is already correct for it -- differing only in
which checkpoint it loads. Nothing about scoring changes, which is the point:
the two ensemble members must differ in weights and nothing else.

WEIGHTS
-------
`ELITEVPR_MIX_WEIGHTS`. Deliberately a separate variable from
`ELITEVPR_WEIGHTS`, so an ensemble run cannot load the same checkpoint twice
and report a "two-model" number that is one model fused with itself -- the
failure mode that would be invisible in the output.

    export ELITEVPR_WEIGHTS=/workspace/model_weights/elitevpr_3ds_s42_e30.pth
    export ELITEVPR_MIX_WEIGHTS=/workspace/ckpt_mixvpr_1024/last_phase1_histogram.pth

REGISTRATION (additive, `elitevpr.py` untouched)
------------------------------------------------
  vpr_models/__init__.py : add `elitevpr_mix` to the import list, and
                           `elif method == "elitevpr_mix": model = elitevpr_mix.get_model()`
  parse.py               : add "elitevpr_mix" to `choices`, and a branch
                           setting descriptors_dimension 1024 / image_size [384,384]
"""
import os

DEFAULT_WEIGHTS = os.environ.get("ELITEVPR_MIX_WEIGHTS", "")


def get_model(weights_path=None, img_size=(384, 384)):
    # Imported inside the call, not at module scope: vpr_models/__init__.py
    # imports every model module at package load, and a top-level sibling
    # import would run during that partially-initialised state.
    from vpr_models.elitevpr import ElitevprModel

    path = weights_path or DEFAULT_WEIGHTS
    if not path:
        raise SystemExit(
            "elitevpr_mix: set ELITEVPR_MIX_WEIGHTS to the second student's "
            "checkpoint. It must NOT be the same file as ELITEVPR_WEIGHTS -- "
            "fusing a model with itself would look like a valid two-model "
            "ensemble in the output.")
    if path == os.environ.get("ELITEVPR_WEIGHTS", ""):
        raise SystemExit(
            f"elitevpr_mix: ELITEVPR_MIX_WEIGHTS and ELITEVPR_WEIGHTS are the "
            f"same file ({path}). That would fuse one model with itself.")
    print(f"[elitevpr_mix] second student: {path}", flush=True)
    return ElitevprModel(path, img_size)
