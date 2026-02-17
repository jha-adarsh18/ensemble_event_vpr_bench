# Ensemble-Based Event Camera Place Recognition Under Varying Illumination

This repository contains the official codebase for the paper:

**Ensemble-Based Event Camera Place Recognition Under Varying Illumination** *Therese Joseph, Tobias Fischer, Michael Milford*

### NSAVP night

<table>
  <tr>
    <td align="center"><b>Event Count</b></td>
    <td align="center"><b>No Polarity</b></td>
    <td align="center"><b>Time Surface</b></td>
    <td align="center"><b>E2VID</b></td>
  </tr>
  <tr>
    <td><img src="./plots/gifs_cropped/R0_RN0_eventCount_0.2_last10.gif" width="220"/></td>
    <td><img src="./plots/gifs_cropped/R0_RN0_eventCount_noPolarity_0.2_last10.gif" width="220"/></td>
    <td><img src="./plots/gifs_cropped/R0_RN0_timeSurface_0.2_last10.gif" width="220"/></td>
    <td><img src="./plots/gifs_cropped/R0_RN0_e2vid_0.2_last10.gif" width="220"/></td>
  </tr>
</table>

---

### NSAVP afternoon

<table>
  <tr>
    <td align="center"><b>Event Count</b></td>
    <td align="center"><b>No Polarity</b></td>
    <td align="center"><b>Time Surface</b></td>
    <td align="center"><b>E2VID</b></td>
  </tr>
  <tr>
    <td><img src="./plots/gifs_cropped/R0_FA0_eventCount_0.2_last10.gif" width="220"/></td>
    <td><img src="./plots/gifs_cropped/R0_FA0_eventCount_noPolarity_0.2_last10.gif" width="220"/></td>
    <td><img src="./plots/gifs_cropped/R0_FA0_timeSurface_0.2_last10.gif" width="220"/></td>
    <td><img src="./plots/gifs_cropped/R0_FA0_e2vid_0.2_last10.gif" width="220"/></td>
  </tr>
</table>

---

### Brisbane daytime

<table>
  <tr>
    <td align="center"><b>Event Count</b></td>
    <td align="center"><b>No Polarity</b></td>
    <td align="center"><b>Time Surface</b></td>
    <td align="center"><b>E2VID</b></td>
  </tr>
  <tr>
    <td><img src="./plots/gifs_cropped/daytime_eventCount_0.2_last10.gif" width="220"/></td>
    <td><img src="./plots/gifs_cropped/daytime_eventCount_noPolarity_0.2_last10.gif" width="220"/></td>
    <td><img src="./plots/gifs_cropped/daytime_timeSurface_0.2_last10.gif" width="220"/></td>
    <td><img src="./plots/gifs_cropped/daytime_e2vid_0.2_last10.gif" width="220"/></td>
  </tr>
</table>

---

### Abstract

Compared to conventional cameras, event cameras provide a high dynamic range and low latency, offering greater robustness to rapid motion and challenging lighting conditions. Although the potential of event cameras for visual place recognition (VPR) has been established, developing robust VPR frameworks under severe illumination changes remains an open research problem.

In this paper, we introduce an **ensemble-based approach** to event camera place recognition that combines sequence-matched results from multiple event-to-frame reconstructions, VPR feature extractors, and temporal resolutions. Our broader fusion strategy delivers significantly improved robustness under varied lighting conditions, achieving a **77% relative improvement in Recall@1** across day-night transitions.

---

## Project Structure

| File | Description |
| --- | --- |
| `load_and_save.py` | Reconstructs event streams into frames using methods like `eventCount`, `timeSurface`, or `e2vid`. |
| `testing.py` | Runs the VPR evaluation pipeline for individual methods and binnings. |
| `evaluate.ipynb` | Jupyter notebook for result aggregation, Recall@N calculation, and plotting. |
| `parser_config.py` | Central configuration for dataset paths and default hyperparameters. |

---

## Requirements

- **Python**: 3.10+
- **OS**: Linux (tested on Linux-64)
- **Dependencies**: Managed via Pixi (see `pixi.toml`)
- **GPU**: Recommended for E2VID reconstruction and deep learning-based VPR methods

---

## Installation & Setup

This project uses [Pixi](https://pixi.sh/) for dependency management.

**1. Install Pixi (if needed)**
```bash
curl -fsSL [https://pixi.sh/install.sh](https://pixi.sh/install.sh) | bash
```

**2. Setup Environment**
Clone the repo and install the dependencies defined in `pixi.toml`.
```bash
git clone https://github.com/theresejoseph/ensemble_event_vpr_bench
cd ensemble_event_vpr_bench/
pixi install
```

## 3. Download Datasets

> **Note:** Download the [**NSAVP**](https://deepblue.lib.umich.edu/data/collections/v118rf157) and [**Brisbane-Event-VPR**](https://huggingface.co/datasets/TobiasRobotics/brisbane-event-vpr/tree/main) datasets from their respective locations for the full experimental suite.

| Dataset | Sensor (DVS) | FOV (H x V) | Resolution |
| :--- | :--- | :--- | :--- |
| **NSAVP** | Inivation DVXplorer | ~70° x 54° | 640 x 480 |
| **Brisbane-Event-VPR** | Inivation DAVIS346 | ~71° x 56° | 346 x 260 |

To quickly test and verify this repository and its methods, a **sample dataset** is available on [Hugging Face](https://huggingface.co/datasets/theresejoseph/datasample_for_ensem_event_bench/) which follows the expected folder structure. 

1. Modify the filepath in `parser_config.py` to point to the datasets to ensure all scripts can run.
2. Use `pixi` to download the dataset via terminal.
```bash
pixi run download-sample-data 
```

## Workflow

### 1. Data Reconstruction

Convert raw event data into visual frames. You can specify different datasets (Brisbane or NSAVP), binning resolutions and reconstruction methods (eventCount, eventCount_noPolarity, itmeSurface, e2vid).

```bash
python load_and_save.py --dataset_type NSAVP --reconstruct_method_name e2vid --time_res 1.0 --dataset_path ./datasample_for_ensem_event_bench/NSAVP/

```

The filename convention is the following:
@{utm_east:.6f}@{utm_north:.6f}@{prefix}_{id}@.jpg

### 2. Individual VPR Evaluation

Run a specific VPR method (e.g., MixVPR, NetVLAD, CosPlace) on the reconstructed frames with a specified seqeunce length (seq_len 1 is no sequence matching) for a reference query pair.

```bash
python testing.py --method mixvpr --dataset_type NSAVP --reconstruct_method_name e2vid --seq_len 10 --ref_seq_idx 9 --qry_seq_idx 10 --dataset_path ./datasample_for_ensem_event_bench/NSAVP/
```

**Sequence Reference IDs:**

Use the following IDs for `--ref_seq_idx` and `--qry_seq_idx`:

| ID | Dataset | Sequence Name | Condition |
| :--- | :--- | :--- | :--- |
| **0** | Brisbane | night | Night |
| **1** | Brisbane | morning | Morning |
| **2** | Brisbane | sunrise | Sunrise |
| **3** | Brisbane | sunset1 | Sunset 1 |
| **4** | Brisbane | sunset2 | Sunset 2 |
| **5** | Brisbane | daytime | Daytime |
| **6** | NSAVP | R0_FA0 | Forward / Afternoon |
| **7** | NSAVP | R0_FS0 | Forward / Sunset |
| **8** | NSAVP | R0_FN0 | Forward / Night |
| **9** | NSAVP | R0_RA0 | Reverse / Afternoon |
| **10** | NSAVP | R0_RS0 | Reverse / Sunset |
| **11** | NSAVP | R0_RN0 | Reverse / Night |

---

### 3. Ensemble Evaluation (Main Contribution)

Combine results from multiple reconstruction methods, VPR feature extractors, and temporal resolutions to achieve robust place recognition under varying illumination.

```bash
python ablate_ensembles.py --dataset_name NSAVP --ref_seq R0_RA0 --qry_seq R0_RN0 --ensemble_over "recon,time" --vpr_methods "mixvpr" --recon_methods "eventCount,timeSurface" --time_strs "0.1,1.0" --seq_len 10 --auto_generate
```

**Parameters:**
- `--ensemble_over`: What to ensemble over (combinations: `recon`, `vpr`, `time`, or any combination)
- `--vpr_methods`: Comma-separated VPR methods (e.g., `mixvpr,cosplace,netvlad`)
- `--recon_methods`: Comma-separated reconstruction methods
- `--time_strs`: Comma-separated temporal resolutions in seconds
- `--seq_len`: Sequence length for temporal matching
- `--auto_generate`: (Optional) Automatically run `testing.py` to generate missing similarity matrices
- `--dataset_path`: (Optional) Override dataset path from config

**Auto-generation of Missing Files:**

If similarity matrices (`.npy` files) don't exist for the requested configurations, the script will detect them and either:
- Exit with a list of missing files (default behavior)
- Automatically generate them by running `testing.py` with appropriate parameters (when `--auto_generate` flag is used)

This saves time by eliminating the need to manually run individual VPR evaluations before ensemble analysis.

---

### 4. Result Analysis & Visualization (`evaluate.ipynb`)

The `evaluate.ipynb` notebook is used to process the CSV outputs generated by the scripts above.

* **Aggregation:** Automatically loads results from the `./results/` directory.
* **Performance Metrics:** Generates Recall@N curves and compares different binning strategies.
* **Visual Debugging:** Visualizes similarity matrices alongside ground truth masks to identify where the localization succeeds or fails across different traverses.

---

### Configuration

The `parser_config.py` file contains default configurations for each dataset. Key parameters include:

- `time_res`: Temporal resolution in seconds (default: 0.1)
- `count_bin`: Enable event count binning (0 or 1)
- `events_per_bin`: Events per bin for count binning (default: 100,000)
- `seq_len`: Sequence length for temporal matching
- `dist_thresh`: Distance threshold in meters for ground truth matching
- Modify `dataset_path` in `default_config` dictionary for Brisbane/NSAVP datasets
- Default: `./datasample_for_ensem_event_bench/NSAVP` (for sample data)

Example Override:
```python
# In parser_config.py, modify dataset_path:
'dataset_path': "/path/to/your/BrisbaneEvent-dataset"
```

---

**Trained Models**

Pre-trained VPR models are located in `trained_models/`:
- MixVPR: `trained_models/mixvpr/resnet50_MixVPR_4096_channels(1024)_rows(4)`
- Models are automatically loaded by the evaluation scripts
- Additional models (NetVLAD, CosPlace) are downloaded automatically on first use

---

**Restuls**

Results are saved to `./results/` with the following naming convention:

```
vpr_results_{dataset}_{binning_type}_{time_res}_{ref_seq}_vs_{qry_seq}_{method}_seq{seq_len}.csv
```

Additionally, raw similarity matrices are saved as `.npy` files in the `./logs/` directory for each experiment. These can be loaded for further analysis or visualization using:
```python
import numpy as np
sim_matrix = np.load('./logs/{experiment_name}/similarity_matrix.npy')
```

---


## Contact & Support
- Open an issue on the GitHub repository
- For dataset-specific questions, refer to the original dataset publications:
  - [NSAVP Dataset](https://deepblue.lib.umich.edu/data/collections/v118rf157)
  - [Brisbane Event VPR Dataset](https://huggingface.co/datasets/TobiasRobotics/brisbane-event-vpr)

- Common issues:
    "Dataset path not found"
    - Verify `dataset_path` in `parser_config.py` points to the correct location
    - Ensure dataset follows expected structure (see sample data)


    "Module not found" errors
    - Ensure Pixi environment is activated: `pixi shell`
    - Reinstall dependencies: `pixi install`

    No matches found / Poor Recall@1"
    - Check ground truth distance threshold (`dist_thresh` in config)
    - Verify GPS data is correctly formatted
    - Try different reconstruction methods or ensemble approaches
    - Inspect similairty matrix for unexpected trends

    "E2VID reconstruction fails"
    - Ensure pre-trained E2VID model is in `e2vid/pretrained/`
    - Check event data format (should be compatible with Tonic library)

---

## Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@ARTICLE{11283034,
  author={Joseph, Therese and Fischer, Tobias and Milford, Michael},
  journal={IEEE Robotics and Automation Letters}, 
  title={Ensemble-Based Event Camera Place Recognition Under Varying Illumination}, 
  year={2026},
  volume={11},
  number={2},
  pages={1290-1297},
  keywords={Feature extraction;Visual place recognition;Cameras;Image reconstruction;Lighting;Reconstruction algorithms;Event detection;Robustness;Pipelines;Surface reconstruction;Localization;computer vision for transportation},
  doi={10.1109/LRA.2025.3641119}}


```

This repository includes code from third-party sources:
- E2VID reconstruction: See [e2vid/LICENSE](e2vid/LICENSE)
- Deep Image Retrieval components: See individual licenses in `vpr_methods_evaluation/third_party/`
