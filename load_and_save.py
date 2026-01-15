import numpy as np
import cv2
import os
import re
import importlib
from pathlib import Path
from parser_config import get_parser, apply_defaults

# Import local dataset classes
from datasets.brisbane_events import BrisbaneEventDataset, Brisbane_RGB_Dataset
from datasets.nsvap import NSVAPDataset, NSAVP_RGB_Dataset

# Set seed for reproducibility
np.random.seed(1)

def args_for_load_save(args_cli=None, reconstruct_method_name=None):
    """
    Initializes arguments and dataset.
    Can be called standalone (no args) or with overrides from testing.py.
    """
    parser = get_parser()
    
    # If called from testing.py with existing args, use them to initialize defaults
    if args_cli is not None:
        # We parse empty list to get defaults, then overlay args_cli values
        args = parser.parse_args([])
        args.dataset_type = args_cli.dataset_type
        args = apply_defaults(args)
        
        # Override specific values
        args.reconstruct_method_name = reconstruct_method_name
        args.adaptive_bin = getattr(args_cli, 'adaptive_bin', args.adaptive_bin)
        args.max_bins = getattr(args_cli, 'max_bins', args.max_bins)
        args.time_res = getattr(args_cli, 'time_res', args.time_res)
        args.count_bin = getattr(args_cli, 'count_bin', args.count_bin)
        args.events_per_bin = getattr(args_cli, 'events_per_bin', args.events_per_bin)
        args.sequences = getattr(args_cli, 'sequences', args.sequences)
        # Ensure dataset_path is carried over if it exists
        if hasattr(args_cli, 'dataset_path'):
            args.dataset_path = args_cli.dataset_path
            
    else:
        # Standalone execution
        args = parser.parse_args()
        args = apply_defaults(args)

    # --- FIX: Set Default dataset_path if missing ---
    if not hasattr(args, 'dataset_path') or args.dataset_path is None:
        # Auto-detect path based on type
        args.dataset_path = f"./datasample_for_ensem_event_bench/{args.dataset_type}"
    # ------------------------------------------------

    # Initialize dataset
    if args.dataset_type.lower() == 'nsavp':
        dataset_class = NSAVP_RGB_Dataset if args.reconstruct_method_name == 'RGB_camera' else NSVAPDataset
    elif args.dataset_type.lower() == 'brisbane':
        dataset_class = Brisbane_RGB_Dataset if args.reconstruct_method_name == 'RGB_camera' else BrisbaneEventDataset
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    dataset = dataset_class(args.dataset_path)

    # Setup Reconstructor
    reconstructor = None
    if args.reconstruct_method_name != 'RGB_camera':
        try:
            module_path = f"reconstruction.{args.reconstruct_method_name}"
            reconstruction_module = importlib.import_module(module_path)
            reconstructor_class = getattr(reconstruction_module, "EventReconstructor")
            reconstructor = reconstructor_class() 
        except ImportError:
            print(f"Warning: Could not import reconstruction module {args.reconstruct_method_name}")

    return args, dataset, reconstructor

def make_paths(args, sequence_name):
    """
    Creates result folders.
    MERGED VERSION:
    1. Uses args.dataset_path (Safe/Flexible) instead of hardcoded '../data'
    2. Keeps the advanced binning logic (Adaptive/Count) needed by run_vpr_save_results
    """
    
    # --- 1. Base Path Safety ---
    # Ensure we have a valid base path. If args.dataset_path is missing, default safely.
    if not hasattr(args, 'dataset_path') or args.dataset_path is None:
         args.dataset_path = f"./datasample_for_ensem_event_bench/{args.dataset_type}"
    
    dataset_base = Path(args.dataset_path)
    recon_root = dataset_base / "image_reconstructions"
    
    # --- 2. Advanced Binning Logic (From New Method) ---
    if args.count_bin == 1 and not getattr(args, 'adaptive_bin', 0):
        # Fixed Count Bins
        bin_type = "fixed"
        subfolder_name = f"{bin_type}_countbins_{args.events_per_bin}"
        filename_base = f"{sequence_name}_{args.reconstruct_method_name}_{args.events_per_bin}{getattr(args, 'exp_tag', '')}"
        
    elif getattr(args, 'adaptive_bin', 0) == 1:
        # Adaptive Bins
        bin_param = f"minres_{args.min_time_res}_maxres{round(args.max_bins*args.min_time_res, 2)}"
        subfolder_name = f"{args.adaptive_bin_tag}_timebins_{args.max_bins}"
        filename_base = f"{sequence_name}_{args.reconstruct_method_name}_{bin_param}{getattr(args, 'exp_tag', '')}"
        
    else:
        # Fixed Time Bins (Default)
        bin_type = "fixed"
        subfolder_name = f"{bin_type}_timebins_{args.time_res}"
        filename_base = f"{sequence_name}_{args.reconstruct_method_name}_{args.time_res}{getattr(args, 'exp_tag', '')}"

    # --- 3. Construct Final Paths ---
    # Directory Structure: /dataset_path/image_reconstructions/fixed_timebins_1.0/e2vid/night/
    images_dir = recon_root / subfolder_name / args.reconstruct_method_name / sequence_name
    processed_path = recon_root / subfolder_name / args.reconstruct_method_name / f"{filename_base}.npz"
    
    # Create the directory
    images_dir.mkdir(parents=True, exist_ok=True)

    # --- 4. Assign Attributes required by run_vpr_save_results ---
    args.save_images_dir = images_dir
    args.video_filename = f"{filename_base}.mp4"
    args.processed_path = processed_path
    
    #run_vpr_save_results needs this string for logging
    args.subfolder_dir = str(recon_root / subfolder_name)

def extract_frame_index(path):
    """Sort helper to ensure frames are ordered by index in videos."""
    match = re.search(r"@frame_(\d+)@", path.name)
    return int(match.group(1)) if match else float('inf')

def save_video(sequence_dir, video_filename):
    """Generates an MP4 preview for visual verification of reconstructions."""
    video_path = Path(sequence_dir).parent / video_filename
    if video_path.exists():
        return

    print(f"Generating video: {video_path}")
    image_paths = sorted(Path(sequence_dir).glob("*.jpg"), key=extract_frame_index)
    if not image_paths: return

    first_img = cv2.imread(str(image_paths[0]))
    if first_img is None: return
    h, w = first_img.shape[:2]
    
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (w, h))
    for img_p in image_paths:
        frame = cv2.imread(str(img_p))
        if frame is not None:
            writer.write(frame)
    writer.release()

def load_save_data(dataset, reconstructor, args, ref_or_qry, return_data=True):
    '''
    Backwards compatible wrapper:
    1. Uses the OLD dataset.process_sequence() method.
    2. Checks if data exists on disk to avoid re-processing.
    3. Reloads data from disk if it already exists and return_data=True.
    '''
    # 1. Resolve sequence name for logging/paths
    seq_idx = args.ref_seq_idx if ref_or_qry == 'ref' else args.qry_seq_idx
    sequence_name = args.sequences[seq_idx]
    
    make_paths(args, sequence_name)
    sequence_dir = Path(args.save_images_dir)
    print(f"Processing sequence: {sequence_name} for {ref_or_qry} from {sequence_dir}", flush=True)

    frames, gt_positions = [], []

    # 2. Check if data already exists
    # Safely check if folder exists and has files
    data_exists = sequence_dir.exists() and any(sequence_dir.iterdir())

    if not data_exists:
        print(f"Processing sequence: {sequence_name}...")

        frames, gt_positions = dataset.process_sequence(args, ref_or_qry, reconstructor)
        
        # Create directory
        sequence_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving {len(frames)} frames to {sequence_dir}")

        # Save Loop
        for i, (frame, pos) in enumerate(zip(frames, gt_positions)):
            # Filename format: @utm_e@utm_n@frame_i@.jpg
            filename = f"@{pos[0]:.6f}@{pos[1]:.6f}@frame_{i}@.jpg"
            filepath = sequence_dir / filename
            
            # Normalize and convert for saving
            if frame.dtype != np.uint8:
                max_val = frame.max()
                frame = (255 * (frame / max_val)) if max_val > 0 else frame
                frame = frame.astype(np.uint8)
            
            # Handle Color conversion (RGB -> BGR for OpenCV)
            img_to_save = frame
            if frame.ndim == 2:
                img_to_save = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 3:
                 img_to_save = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(filepath), img_to_save)
            
        print(f"Saved {len(frames)} new images.")
        
    elif return_data:
        # --- PATH B: RELOAD FROM DISK ---
        print(f"Data found. Loading frames from {sequence_dir}...")
        
        # Helper to sort by frame index
        def extract_frame_index(p):
            try:
                # Extracts '10' from '...frame_10@.jpg'
                return int(str(p.name).split('@')[3].replace('frame_', ''))
            except (IndexError, ValueError):
                return -1

        image_paths = sorted(list(sequence_dir.glob("*.jpg")) + list(sequence_dir.glob("*.png")), key=extract_frame_index)
        
        frames_list, pos_list = [], []
        
        for img_path in image_paths:
            parts = img_path.name.split('@')
            if len(parts) >= 4:
                try:
                    # Parse filename for GPS
                    utm_east = float(parts[1])
                    utm_north = float(parts[2])
                    pos_list.append(np.array([utm_east, utm_north]))

                    # Load Image
                    frame = cv2.imread(str(img_path))
                    if frame is None:
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Back to RGB
                    frames_list.append(frame)
                except ValueError:
                    continue

        frames = np.stack(frames_list) if frames_list else np.empty((0,))
        gt_positions = np.stack(pos_list) if pos_list else np.empty((0, 2))
        print(f"Loaded {len(frames)} frames and positions from disk.")

    else:
        # Data exists, but return_data is False (e.g. bulk processing only)
        print(f"Skipping {sequence_name}: Data already processed and return_data=False.")

    # --- Save video (Common to both paths) ---
    if getattr(args, 'save_frames_video', False):
        try:
            # Assuming save_video is imported or available
            save_video(sequence_dir, args.video_filename)
        except NameError:
             print("Warning: save_video function not defined/imported.")
    
    return frames, gt_positions

def process_and_save_sequence(dataset, reconstructor, args, sequence_name):
    """Processes a single traverse and saves images with GPS/UTM metadata."""
    make_paths(args, sequence_name)
    sequence_dir = args.save_images_dir

    # Skip if folder already has images to allow for resuming interrupted jobs
    if any(sequence_dir.iterdir()):
        print(f"Skipping {sequence_name}: Data already processed.")
        return

    print(f"Processing sequence: {sequence_name}...")
    
    # We pass the sequence_name directly to the dataset processor
    frames, gt_positions = dataset.process_sequence_by_name(args, sequence_name, reconstructor)
    
    print(f"Saving {len(frames)} frames to {sequence_dir}")
    for i, (frame, pos) in enumerate(zip(frames, gt_positions)):
        # Filename format: @utm_e@utm_n@frame_i@.jpg
        filename = f"@{pos[0]:.6f}@{pos[1]:.6f}@frame_{i}@.jpg"
        filepath = sequence_dir / filename
        
        # Normalize and convert to 8-bit for saving
        if frame.dtype != np.uint8:
            frame = (255 * (frame / (frame.max() if frame.max() > 0 else 1))).astype(np.uint8)
        
        img_to_save = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame
        cv2.imwrite(str(filepath), img_to_save)
    
    # Trigger video save based on parser_config 'save_frames_video' key
    if getattr(args, 'save_frames_video', False):
        save_video(sequence_dir, args.video_filename)

if __name__ == "__main__":
    args, dataset, reconstructor = args_for_load_save()
    
    print(f"Starting Bulk Reconstruction for {args.dataset_type} Dataset")
    
    # Iterate through every sequence defined in your parser_config
    for sequence_name in args.sequences:
        try:
            process_and_save_sequence(dataset, reconstructor, args, sequence_name)
        except Exception as e:
            print(f"Critical error processing {sequence_name}: {e}")
            
