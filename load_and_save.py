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

def args_for_load_save():
    """Initializes arguments and dataset based on parser_config settings."""
    parser = get_parser()
    args = parser.parse_args()
    args = apply_defaults(args)

    # Initialize dataset based on the path provided in parser_config
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
        module_path = f"reconstruction.{args.reconstruct_method_name}"
        reconstruction_module = importlib.import_module(module_path)
        reconstructor_class = getattr(reconstruction_module, "EventReconstructor")
        reconstructor = reconstructor_class() 

    return args, dataset, reconstructor

def make_paths(args, sequence_name):
    """Creates a result folder within the dataset_path for processed frames."""
    dataset_base = Path(args.dataset_path)
    recon_root = dataset_base / "image_reconstructions"
    
    # Simplified binning logic for RA-L experiments
    if args.count_bin == 1:
        subfolder_name = f"fixed_countbins_{args.events_per_bin}"
        filename_base = f"{sequence_name}_{args.reconstruct_method_name}_{args.events_per_bin}{args.exp_tag}"
    else:
        subfolder_name = f"fixed_timebins_{args.time_res}"
        filename_base = f"{sequence_name}_{args.reconstruct_method_name}_{args.time_res}{args.exp_tag}"

    # Directory Structure: /dataset_path/image_reconstructions/fixed_timebins_0.1/e2vid/night/
    images_dir = recon_root / subfolder_name / args.reconstruct_method_name / sequence_name
    images_dir.mkdir(parents=True, exist_ok=True)

    args.save_images_dir = images_dir
    args.video_filename = f"{filename_base}.mp4"

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
            
