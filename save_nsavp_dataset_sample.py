import h5py
import os
import numpy as np
import shutil

# Configuration

base_dir = "/work/qvpr/data/processed/NSAVP-ensemble_bench/h5_data/"
out_dir = "/work/qvpr/data/processed/NSAVP-ensemble_bench/h5_data_sample"
os.makedirs(out_dir, exist_ok=True)

sequence = "R0_FA0"
sample_duration_ns = 20 * 1e9  # 20 seconds


def h5_bisect_left(dataset, value):
    """
    Performs a binary search on an HDF5 dataset WITHOUT loading it into memory.
    Reads only ~30 single values instead of the full array.
    """
    low, high = 0, dataset.shape[0]
    while low < high:
        mid = (low + high) // 2
        # Read only the single middle value from disk
        if dataset[mid] < value:
            low = mid + 1
        else:
            high = mid
    return low

def create_fast_sample(seq_name):
    evt_in = os.path.join(base_dir, f"{seq_name}_dvxplorer_left.h5")
    gps_in = os.path.join(base_dir, f"{seq_name}_applanix.h5")
    
    evt_out = os.path.join(out_dir, f"{seq_name}_dvxplorer_left.h5")
    gps_out = os.path.join(out_dir, f"{seq_name}_applanix.h5")

    print(f"Processing {seq_name}...")

    # --- 1. Fast GPS Movement Detection (Vectorized) ---
    start_time_ns = None
    
    with h5py.File(gps_in, "r") as f:
        # GPS files are small enough to load fully (usually <100MB)
        gps_t = f["pose_base_link/timestamps"][:]
        gps_pos = f["pose_base_link/positions"][:, :2]

        # Vectorized check: Calculate ALL distances at once (100x faster than loops)
        # Equivalent to your loop: checks distance between i and i+100
        deltas = gps_pos[100:] - gps_pos[:-100]
        distances = np.linalg.norm(deltas, axis=1)
        
        # Find the first index where distance > 1.0 meter
        moving_indices = np.where(distances > 1.0)[0]
        
        if moving_indices.size > 0:
            start_idx = moving_indices[0]
            start_time_ns = gps_t[start_idx]
            print(f"  ✅ Detected movement starting at t={start_time_ns} ns")
        else:
            print("  ❌ No significant movement found!")
            return

    end_time_ns = start_time_ns + sample_duration_ns

    # --- 2. Fast Event Slicing (No Full Read) ---
    with h5py.File(evt_in, "r") as src, h5py.File(evt_out, "w") as dst:
        dst.attrs.update(src.attrs)
        t_dset = src["events/timestamps"]

        print("  Events: Finding start/end indices (smart search)...")
        
        # Custom binary search that reads directly from disk
        # This replaces np.searchsorted(t_dset) which causes the massive RAM spike
        start_idx = h5_bisect_left(t_dset, start_time_ns)
        end_idx = h5_bisect_left(t_dset, end_time_ns)
        
        num_events = end_idx - start_idx
        print(f"  Events: Slicing {num_events} events...")

        for key in ["timestamps", "x_coordinates", "y_coordinates", "polarities"]:
            path = f"events/{key}"
            if path in src:
                # Only read the specific slice we need into memory
                data = src[path][start_idx:end_idx]
                dset = dst.create_dataset(path, data=data, compression="gzip")
                dset.attrs.update(src[path].attrs)

    # --- 3. Copy Full GPS ---
    print(f"  GPS: Copying full file...")
    shutil.copy2(gps_in, gps_out)
    print(f"✅ Fast sample complete: {out_dir}")

create_fast_sample(sequence)
