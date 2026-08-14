"""testing.py with stationary-frame removal, per scripts/evaluate_brisbane.py.

WHAT IS DIFFERENT FROM testing.py
---------------------------------
Exactly one thing: frames below `--min_speed_ms` are dropped from BOTH the
reference and the query side before matching, and the pair is scored twice --
`[all]` (every frame, i.e. what testing.py already reports) and `[moving]`
(stops removed). Everything else -- reconstruction, descriptor extraction,
the cached similarity matrix, `seq_match_row`, `getPRCurve_sim`, the 25 m
threshold -- is the pipeline's own code, imported rather than copied.

The mask is `scripts/evaluate_brisbane.py:346-351` verbatim in behaviour:

    v = ||p[i+1] - p[i]|| / step ;  v = concat([v, v[-1:]]) ;  keep v >= min_speed

with `step = args.time_res`, because the bench emits one frame per time bin.

WHY THE FILTER GOES WHERE IT DOES
---------------------------------
`compute_similarity_matrices` is called on the FULL traverses and its cache
(`logs/<log_dir>/<ref>_vs_<qry>_...npy`) is keyed on method/rep/time_res only.
Filtering after it means this script reuses testing.py's cached matrices and
cannot poison them -- the descriptors it scores are byte-identical to the ones
testing.py scores. Only rows/columns are dropped afterwards.

Ground truth is recomputed, not remapped: `TestDataset.get_positives()` is
re-run on the filtered UTMs, so positive indices refer to filtered database
columns. That reuses the bench's own KNN rather than re-deriving the GT.

KNOWN PROPERTY, STATED PLAINLY (same as evaluate_brisbane.py)
------------------------------------------------------------
Sequence matching runs over the KEPT frames, so an L=10 window spans more real
time in `[moving]` than in `[all]`. That is what removing stops means for a
slope-1 matcher and it is what evaluate_brisbane.py does; it is not a bug, but
`[moving]` and `[all]` are therefore not the same protocol. The ensemble
baseline does NOT remove stationary frames, so `[all]` stays the
protocol-matched column for any comparison against Table 1.

USAGE -- identical to testing.py plus one flag
----------------------------------------------
    python testing_moving.py --method elitevpr \
        --reconstruct_method_name eliteHistogram --time_res 1.0 \
        --seq_len 10 --ref_seq_idx 6 --qry_seq_idx 7 [--min_speed_ms 1.0]

Both variants print a line carrying `AUC=` and `Recall at 1:`, so the grep in
the existing sweep still works -- it just returns two rows per run now, `[all]`
first.
"""
import argparse
import sys
from glob import glob
from pathlib import Path

import numpy as np

from testing import default_args_testing, args_for_vpr, process_pair
from vpr_methods_evaluation.main import (compute_similarity_matrices,
                                         plot_simmat_with_matches,
                                         getPRCurve_sim)
from vpr_methods_evaluation.test_dataset import TestDataset


def moving_mask(pos, step, min_speed):
    """scripts/evaluate_brisbane.py:346-351, in numpy.

    Forward difference with the last value repeated, so the mask is the same
    length as the traverse and the final frame inherits its predecessor's
    speed instead of being dropped for having no successor.
    """
    if len(pos) < 2:
        return np.ones(len(pos), dtype=bool)
    v = np.linalg.norm(pos[1:] - pos[:-1], axis=1) / step
    v = np.concatenate([v, v[-1:]])
    return v >= min_speed


def _check_temporal_order(paths, side):
    """The speed above is a difference between ADJACENT ROWS, which is only a
    speed if the rows are in time order. `read_images_paths` sorts on the
    `@frame_XXX@` index, but it skips that sort when a
    `<folder>_images_paths.txt` exists. Fail loudly rather than difference
    frames that are out of order."""
    try:
        idx = [int(p.split("@")[-2].split("_")[1]) for p in paths]
    except (IndexError, ValueError):
        print(f"  ! {side}: cannot read @frame_N@ from the filenames; "
              f"assuming the listed order is temporal")
        return
    if any(b <= a for a, b in zip(idx, idx[1:])):
        raise RuntimeError(
            f"{side} frames are not in increasing @frame_N@ order, so "
            f"consecutive differences are not speeds. Delete the stale "
            f"'<folder>_images_paths.txt' next to the image folder and re-run.")


def run_vpr_moving(args, min_speed):
    """vpr_methods_evaluation.main.run_vpr with the stationary filter added.

    Returns {variant: (recall_at_1, auc)}.
    """
    from load_and_save import make_paths

    metric = "l2"
    args.num_patches = (args.patch_num_cols * args.patch_num_rows
                        if args.grid_or_nest == "grid" else args.num_patches)
    seq_len = args.seq_len
    args.positive_dist_threshold = 25
    args.rep = args.reconstruct_method_name
    ref_seq = args.sequences[args.idR]
    qry_seq = args.sequences[args.idQ]
    args.dataset_type = "NSAVP" if 6 <= args.idR <= 11 else "Brisbane"

    make_paths(args, ref_seq)
    args.database_folder = str(args.save_images_dir)
    num_ref_frames = len(glob(f"{args.database_folder}/**/*", recursive=True))
    make_paths(args, qry_seq)
    args.queries_folder = str(args.save_images_dir)
    num_qry_frames = len(glob(f"{args.queries_folder}/**/*", recursive=True))
    if args.save_images_dir is None:
        print(f"Failed to load data for {ref_seq} vs {qry_seq}")
        return None
    args.log_dir = args.subfolder_dir.split("/")[-1]

    print("Loaded data:")
    print(f"  Reference: {args.database_folder} ({num_ref_frames} frames)")
    print(f"  Query: {args.queries_folder} ({num_qry_frames} frames)")
    print(f"  Log directory: {args.log_dir}")

    log_dir = Path("logs") / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    simMatPath = (log_dir / f"{ref_seq}_vs_{qry_seq}_{args.method}_{metric}"
                            f"_reconstruct_{args.reconstruct_method_name}"
                            f"_{args.time_res}_{args.patch_or_frame}.npy")
    print(f"Similarity matrix path: {simMatPath}")
    S, _sim_time = compute_similarity_matrices(simMatPath, args, metric)
    if S is None:
        print("⚠ Skipping due to invalid similarity matrix")
        return None

    test_ds = TestDataset(args.database_folder, args.queries_folder,
                          positive_dist_threshold=args.positive_dist_threshold,
                          image_size=args.image_size, use_labels=args.use_labels)
    _check_temporal_order(test_ds.database_paths, "reference")
    _check_temporal_order(test_ds.queries_paths, "query")

    db_utms = np.asarray(test_ds.database_utms, dtype=float)
    q_utms = np.asarray(test_ds.queries_utms, dtype=float)

    # One frame per time bin, so the sample period IS time_res. Count binning
    # makes the period frame-dependent, at which case a fixed-step speed is
    # meaningless -- refuse instead of reporting a wrong filter.
    step = float(args.time_res)
    if int(getattr(args, "count_bin", 0)):
        raise SystemExit(
            "--count_bin=1 gives a non-uniform frame period, so a fixed-step "
            "speed cannot be computed. Run the moving variant with time "
            "binning (count_bin=0).")

    out = {}
    for variant in ("all", "moving"):
        if variant == "moving":
            r_keep = moving_mask(db_utms, step, min_speed)
            q_keep = moving_mask(q_utms, step, min_speed)
            print(f"\n  [moving] frames kept at >= {min_speed} m/s: "
                  f"{{'{ref_seq}': '{int(r_keep.sum())}/{len(r_keep)}', "
                  f"'{qry_seq}': '{int(q_keep.sum())}/{len(q_keep)}'}}")
            if r_keep.sum() < 2 or q_keep.sum() < 2:
                print("  [moving] fewer than 2 frames survive; skipping")
                continue
        else:
            r_keep = np.ones(len(db_utms), dtype=bool)
            q_keep = np.ones(len(q_utms), dtype=bool)

        # Recompute GT on the filtered traverses so positive indices address
        # filtered database columns.
        test_ds.database_utms = db_utms[r_keep]
        test_ds.queries_utms = q_utms[q_keep]
        positives_per_query = test_ds.get_positives()

        if args.patch_or_frame == "patch":
            num_queries, num_refs = S[0][q_keep][:, r_keep].shape
            combined = np.zeros((num_queries, num_refs), dtype=np.float32)
            for j in range(args.num_patches):
                seqMat_patch, _ = plot_simmat_with_matches(
                    S[j][q_keep][:, r_keep], positives_per_query, None,
                    patch_num=f"_Patch{j}-{args.num_patches}", seq_len=seq_len,
                    seq_match_type=args.seq_match_type)
                combined += np.array(seqMat_patch)
            _, all_matches = plot_simmat_with_matches(
                combined, positives_per_query, None, patch_num="patch_combined",
                seq_len=1, seq_match_type=args.seq_match_type)
            _, auc, _r1 = getPRCurve_sim(np.array(combined), positives_per_query)
        else:
            seqMat, all_matches = plot_simmat_with_matches(
                np.asarray(S)[q_keep][:, r_keep], positives_per_query, None,
                patch_num="frame", seq_len=seq_len,
                seq_match_type=args.seq_match_type)
            _, auc, _r1 = getPRCurve_sim(np.array(seqMat), positives_per_query)

        correct = sum(best in positives_per_query[q]
                      for q, best, _, _ in all_matches)
        recall_at_1 = correct / len(all_matches) if all_matches else 0.0
        out[variant] = (recall_at_1, auc)
        print(f"[{variant:6s}] {ref_seq} vs {qry_seq}, method={args.method}, "
              f"rep={args.rep}, seq_len={seq_len}, AUC={auc:.4f}")
        print(f"[{variant:6s}] Recall at 1: {recall_at_1}")

    return out


def main():
    # Pull our own flag out of argv first: default_args_testing() owns the rest
    # of the parser and testing.py is not edited.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--min_speed_ms", type=float, default=1.0,
                     help="drop frames slower than this, both sides "
                          "(0 disables, giving plain testing.py behaviour)")
    mine, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    args_cli = default_args_testing()
    idR, idQ = args_cli.ref_seq_idx, args_cli.qry_seq_idx
    recon = args_cli.reconstruct_method_name
    print(f"Running VPR for idR={idR}, idQ={idQ}, "
          f"reconstruct_method_name={recon}, min_speed_ms={mine.min_speed_ms}")

    recon, idR, idQ, _lens = process_pair(args_cli, recon, idR, idQ)
    args_vpr = args_for_vpr(args_cli, recon, idR, idQ)
    run_vpr_moving(args_vpr, mine.min_speed_ms)


if __name__ == "__main__":
    main()
