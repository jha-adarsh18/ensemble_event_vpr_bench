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
### 2. Installation & Setup

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

**3. Download Datsets**
Note: download the [**NSAVP**](https://deepblue.lib.umich.edu/data/collections/v118rf157) and [**BrisbaneEvent**](https://huggingface.co/datasets/TobiasRobotics/brisbane-event-vpr/tree/main) datasets from their respective locations for the full experimental suite.

To quickly test and verify this repository and its methods, a **sample dataset** is available on [Hugging Face](https://huggingface.co/datasets/theresejoseph/datasample_for_ensem_event_bench/) which follows the expected folder structure. Modify the filepath in `parser_config.py` to point to the datasets to ensure all scripts can run. Use pixi to download the dataset via terminal. 

```bash
pixi run download-sample-data 
```

## Workflow

### 1. Data Reconstruction

Convert raw event data into visual frames. You can specify different datasets (Brisbane or NSAVP), binning resolutions and reconstruction methods (eventCount, eventCount_noPolarity, itmeSurface, e2vid).

```bash
python load_and_save.py --dataset_type NSAVP --reconstruct_method_name e2vid --time_res 1.0

```

The filename convention is the following:
@{utm_east:.6f}@{utm_north:.6f}@{prefix}_{id}@.jpg

### 2. Individual VPR Evaluation

Run a specific VPR method (e.g., MixVPR, NetVLAD, CosPlace) on the reconstructed frames with a specified seqeunce length (seq_len 1 is no sequence matching) for a reference query pair.

```bash
python testing.py --method mixvpr --dataset_type NSAVP --reconstruct_method_name e2vid --seq_len 10 --ref_seq_idx 9 --qry_seq_idx 10
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

### 3. Result Analysis & Visualization (`evaluate.ipynb`)

The `evaluate.ipynb` notebook is used to process the CSV outputs generated by the scripts above.

* **Aggregation:** Automatically loads results from the `./results/` directory.
* **Performance Metrics:** Generates Recall@N curves and compares different binning strategies.
* **Visual Debugging:** Visualizes similarity matrices alongside ground truth masks to identify where the localization succeeds or fails across different traverses.

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
