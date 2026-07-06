# [Manuscript title]: A biomechanical wrist interface for continuous finger-level interaction

[![Paper](https://img.shields.io/badge/Paper-Nature%20Portfolio-lightgrey)]([ARTICLE_URL])
[![Code DOI](https://img.shields.io/badge/Code%20DOI-[DOI]-blue)]([CODE_DOI_URL])
[![Data DOI](https://img.shields.io/badge/Data%20DOI-[DOI]-green)]([DATA_DOI_URL])
[![License](https://img.shields.io/badge/License-[LICENSE]-informational)](LICENSE)

This repository contains the training and real-time inference code used for **four-class hand-gesture/cursor-control recognition** from two wearable sensor channels. The pipeline preprocesses resistance/sensor traces into smoothed sliding windows, trains an ultra-light one-dimensional convolutional neural network (1-D CNN), and deploys the trained model through a Bluetooth Low Energy (BLE) real-time interface.

The code supports the following workflow:

1. **Offline preprocessing** of labelled sensor recordings stored as CSV files.
2. **Model training and validation** using an ultra-light 1-D CNN.
3. **Real-time BLE inference** with calibration, min–max normalization, live signal monitoring, CSV logging, and cursor-feedback control.

> **Publication note.** Before public release, replace all placeholders in this README, archive the final code release with a persistent identifier such as a DOI, add a licence, and provide either the raw/processed data or a precise data-access statement. A GitHub link alone is not sufficient for long-term reproducibility.

---

## Contents

- [Repository structure](#repository-structure)
- [System requirements](#system-requirements)
- [Installation](#installation)
- [Input data](#input-data)
- [Configuration](#configuration)
- [Offline model training](#offline-model-training)
- [Real-time BLE inference](#real-time-ble-inference)
- [Model architecture](#model-architecture)
- [Outputs](#outputs)
- [Reproducibility notes](#reproducibility-notes)
- [Troubleshooting](#troubleshooting)
- [Data, code and model availability](#data-code-and-model-availability)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Repository structure

```text
.
├── README.md                         # Repository documentation
├── params.py                         # Paths, preprocessing parameters and training hyperparameters
├── preprocessing_train.py            # Dataset class for labelled CSV recordings
├── train.py                          # Offline model training and validation script
├── ultralight_model.py               # Ultra-light 1-D CNN model definition
├── preprocessing_main_exp.py          # BLE packet parsing and real-time preprocessing dataset
├── Application2_main_exp.py           # Real-time BLE application, GUI, logging and cursor control
├── train/                            # Labelled training CSV files; user-provided
├── valid/                            # Labelled validation CSV files; user-provided
├── test/                             # Optional held-out test CSV files; user-provided
├── model_log/                        # Trained model checkpoints; created by the user/script
├── background.png                    # GUI background image; required for real-time application
├── cursor.png                        # GUI cursor image; required for real-time application
└── chrome.png                        # GUI pinch-feedback image; required for real-time application
```

The listed directory layout is the expected public-release layout. The uploaded code currently contains hard-coded output filenames and experiment-specific paths. For a clean public repository, keep these paths configurable through `params.py` or command-line arguments rather than editing source files for each run.

---

## System requirements

### Hardware

Offline training can run on either CPU or CUDA-enabled GPU. Real-time inference requires:

- A BLE-enabled computer.
- A compatible multi-channel BLE sensor device.
- Access to the BLE characteristic used by the device.
- GUI support for Tkinter.

The current real-time script is configured for a device named `MK_SS_8channels` and uses the characteristic UUID:

```text
00002a37-0000-1000-8000-00805f9b34fb
```

Update the BLE device address and selected channels before running the real-time application.

### Software

The code was written for Python 3 and uses PyTorch. A Python 3.9 environment is recommended unless the accompanying paper reports a different tested version.

Required Python packages:

```text
attrs
bleak
numpy
pandas
Pillow
torch
tqdm
```

Tkinter is also required. It is included with many Python distributions, but on some Linux systems it must be installed separately through the system package manager.

---

## Installation

Create a clean environment:

```bash
conda create -n gesture-cnn python=3.9
conda activate gesture-cnn
```

Install PyTorch using the command appropriate for your operating system and CUDA version from the official PyTorch installation selector. Then install the remaining dependencies:

```bash
pip install attrs bleak numpy pandas Pillow tqdm
```

For a publication archive, include a version-locked `environment.yml` or `requirements.txt`. Without pinned versions, exact reproduction of the reported numbers is weaker than it needs to be.

A minimal `requirements.txt` would be:

```text
attrs
bleak
numpy
pandas
Pillow
torch
tqdm
```

---

## Input data

### Labelled CSV files for offline training

`preprocessing_train.py` expects each training and validation file to be a CSV file with the following columns:

| Column name | Description |
|---|---|
| `Counts` | Integer time/sample index. Windows with non-consecutive indices are skipped. |
| `GestreIdentifier` | Gesture label used for window-level class assignment. Keep this spelling unless the code is updated. |
| `Channel 2` | Sensor channel used as `sensor1` in the current training pipeline. |
| `Channel 3` | Sensor channel used as `sensor2` in the current training pipeline. |

Place files in the directories configured in `params.py`:

```text
train/
valid/
test/      # optional unless a separate test script is added
```

### Label mapping

The current preprocessing code maps raw gesture identifiers to four model classes:

| Raw gesture identifier | Model class | Meaning |
|---:|---:|---|
| `0` | `0` | Rest |
| `9` | `1` | Left |
| `3` | `2` | Right |
| `20` | `3` | Pinch |
| `30` | — | Ignored during data loading |

A window is assigned a class only when the required number of samples in that window meet the corresponding threshold. Windows that do not satisfy any class threshold are excluded.

### Real-time BLE packets

`preprocessing_main_exp.py` parses BLE packets by removing the first byte of the payload, splitting the remainder into 16-bit words, swapping the byte order of each word, and extracting the channels listed in `params.selected_channels`. Parsed channels are stored internally as `Sensor 1`, `Sensor 2`, and so on according to the order in `selected_channels`.

---

## Configuration

All core parameters are defined in `params.py`:

```python
train_dir = "train"
validate_dir = "valid"
test_dir = "test"
model_dir = "model_log"

window_size = 120
label_threshold = 120
pinch_threshold = 120
stride = 1
num_to_ignore = 3
early_stopping_threshold = 85
selected_channels = (2, 1)

lr = 0.0001
batch_size = 8
num_workers = 4
num_epoch = 200
```

Important parameters:

| Parameter | Meaning |
|---|---|
| `window_size` | Number of raw samples in each sliding window. |
| `stride` | Step size for offline sliding-window generation. |
| `num_to_ignore` | Number of extreme values removed from each sorted chunk during trimmed-mean smoothing. |
| `label_threshold` | Minimum number of samples needed to assign rest/left/right class to a window. |
| `pinch_threshold` | Minimum number of samples needed to assign pinch class to a window. |
| `selected_channels` | BLE channels used by the real-time pipeline, in extraction order. |
| `early_stopping_threshold` | Validation accuracy threshold at which training stops early. |

The offline and real-time pipelines must use compatible preprocessing settings. In particular, do not deploy a checkpoint trained with one `window_size` using a real-time script configured with a different `window_size`.

---

## Offline model training

### 1. Prepare directories

```bash
mkdir -p train valid test model_log
```

Place labelled CSV files into `train/` and `valid/`.

### 2. Verify the input format

Each CSV file must contain:

```text
Counts, GestreIdentifier, Channel 2, Channel 3
```

The `Counts` column should increase by one between consecutive samples. The preprocessing script rejects windows containing discontinuities in `Counts`.

### 3. Train the model

```bash
python train.py
```

During training, the script:

1. Loads all files from `params.train_dir` and `params.validate_dir`.
2. Applies per-sensor min–max normalization.
3. Generates sliding windows.
4. Splits each window into chunks of 10 samples.
5. Computes a trimmed mean for each chunk.
6. Trains the 1-D CNN using cross-entropy loss and Adam optimization.
7. Evaluates validation accuracy after each epoch.
8. Saves the best checkpoint to `model_log/`.

### 4. Expected checkpoint output

The current script saves checkpoints using hard-coded filenames similar to:

```text
model_log/App2_model_0405data_ws90_for_demo_0416_2ch_4out_until_85acc.pth
model_log/App2_model_<accuracy>_<epoch>_0405data_ws90_for_demo_0416_2ch_4out_until_85acc.pth
```

For public release, replace these experiment-specific names with descriptive, reproducible filenames such as:

```text
model_log/ultralightcnn1d_window120_channels2-1_seed0_best.pth
```

---

## Real-time BLE inference

`Application2_main_exp.py` implements the real-time application. It receives BLE packets, preprocesses them into model inputs, performs inference, displays live sensor traces, logs selected channels to CSV, and updates a cursor-control interface.

### Required files

Before running the real-time application, ensure the following are present:

```text
background.png
cursor.png
chrome.png
model_log/<trained_model_checkpoint>.pth
```

### Update module imports before public release

The uploaded real-time script currently imports legacy module names:

```python
from App2_model_ultralight_2in_4out_all import UltraLightCNN1D
from App2_dataset_2in_realtime_demo import BLEPacketDataset
from App2_params import Params
```

If the repository uses the filenames provided here, update those lines to:

```python
from ultralight_model import UltraLightCNN1D
from preprocessing_main_exp import BLEPacketDataset
from params import Params
```

Do this before sharing the repository. Otherwise, a new user will not be able to run the application from a clean clone.

### Configure BLE and model paths

In `Application2_main_exp.py`, update:

```python
DEVICE_ADDRESS = "<YOUR_BLE_DEVICE_ADDRESS>"
model_path = "model_log/<YOUR_TRAINED_MODEL>.pth"
```

Also verify:

```python
REST_PROB_THRESHOLD = 0.80
PINCH_PROB_THRESHOLD = 0.85
PINCH_COOLDOWN_SECONDS = 3.0
```

These thresholds determine when low-confidence predictions are reassigned to rest and how often pinch feedback can be triggered.

### Run real-time inference

```bash
python Application2_main_exp.py
```

The application sequence is:

1. Connect to the BLE device.
2. Display a waiting screen.
3. Collect normalization data for the selected channels.
4. Compute per-channel min–max values.
5. Load the trained model checkpoint.
6. Start the real-time inference loop.
7. Display live sensor values and cursor-feedback state.

### Prediction-to-control mapping

| Predicted class | Meaning | Real-time action |
|---:|---|---|
| `0` | Rest | No cursor movement |
| `1` | Left | Move cursor left |
| `2` | Right | Move cursor right |
| `3` | Pinch | Show pinch feedback image with cooldown |

---

## Model architecture

`ultralight_model.py` defines `UltraLightCNN1D`, a compact 1-D CNN for two-channel time-series classification.

Input shape:

```text
(batch_size, 2, T)
```

where `T = window_size / 10` when `window_size` is divisible by the smoothing chunk size of 10.

Architecture:

```text
Conv1d(2 → 16, kernel_size=3, padding=1)
BatchNorm1d(16)
ReLU
MaxPool1d(kernel_size=2)
Conv1d(16 → 32, kernel_size=3, padding=1)
BatchNorm1d(32)
ReLU
AdaptiveAvgPool1d(1)
Flatten
Linear(32 → 4)
```

The model outputs logits for four classes: rest, left, right and pinch.

---

## Outputs

### Training outputs

| Output | Description |
|---|---|
| `model_log/*.pth` | PyTorch model checkpoint files. |
| `accuracy_*.csv` | Validation accuracy values over training epochs. The current filename is hard-coded in `train.py`. |

### Real-time outputs

| Output | Description |
|---|---|
| `realtime_csv_logs_*/selected_channels_<timestamp>.csv` | Real-time selected-channel sensor log. |
| GUI control window | Cursor-feedback visualization. |
| GUI monitor window | Live sensor traces and min–max guide lines. |

---

## Reproducibility notes

This code can support a reproducible computational workflow, but the following items must be addressed before a Nature Portfolio submission or public release:

1. **Provide the data or state access conditions.** The repository needs the training/validation/test data, a DOI-linked data repository, or a clear data-access statement explaining restrictions.
2. **Archive the exact code version.** Create a tagged release and deposit it in a repository that assigns a DOI, such as Zenodo or Code Ocean.
3. **Add a software licence.** Without a licence, reuse is legally ambiguous.
4. **Pin software versions.** Add `environment.yml`, `requirements.txt`, or a container recipe.
5. **Remove private hardware identifiers.** Replace BLE MAC addresses and experiment-specific filenames with placeholders or configuration values.
6. **Fix legacy imports.** Align the real-time application imports with the public filenames.
7. **Report split strategy.** Document how recordings were divided into train, validation and test sets. Do not mix windows from the same continuous trial across splits unless this is explicitly justified.
8. **Set random seeds for exact repeatability.** The current training script shuffles data but does not set a seed. Report whether results are averaged across repeated runs.
9. **Use an independent test set.** Validation accuracy is useful for model selection; manuscript performance claims should be based on a held-out test set or another appropriate evaluation protocol.
10. **Keep preprocessing consistent.** The `window_size`, smoothing chunk size, channel order and normalization procedure must match between training and deployment.

---

## Troubleshooting

### `ModuleNotFoundError` in the real-time application

Update the legacy imports in `Application2_main_exp.py` to match the public filenames:

```python
from ultralight_model import UltraLightCNN1D
from preprocessing_main_exp import BLEPacketDataset
from params import Params
```

### Empty training or validation dataset

Check that:

- CSV files are present in `train/` and `valid/`.
- Column names match exactly, including `GestreIdentifier`.
- `Counts` values are consecutive within windows.
- Label thresholds are not too strict for the available data.

### Model shape mismatch during real-time inference

Confirm that the deployed model was trained with the same:

- Number of channels.
- Channel order.
- `window_size`.
- Smoothing chunk size.
- Number of output classes.

### BLE connection failure

Check that:

- The BLE device is powered and advertising.
- `DEVICE_ADDRESS` is correct for the current device.
- The characteristic UUID matches the sensor firmware.
- No other program is connected to the same BLE device.

### Tkinter or image-loading error

Ensure the following image files are in the working directory:

```text
background.png
cursor.png
chrome.png
```

On Linux, install Tkinter if it is not included with the Python distribution.

---

## Data, code and model availability

Replace this section with the final availability statements used in the manuscript.

**Code availability.** The source code used for preprocessing, model training and real-time BLE inference is available at `[REPOSITORY_URL]` and archived at `[CODE_DOI_URL]` with DOI `[CODE_DOI]`.

**Data availability.** The labelled sensor recordings and any processed datasets required to reproduce the reported results are available at `[DATA_REPOSITORY_URL]` with DOI/accession `[DATA_DOI_OR_ACCESSION]`. If data cannot be shared openly, state the reason, access conditions, contact address, expected response time and any restrictions on reuse.

**Model availability.** Trained model checkpoints are available at `[MODEL_REPOSITORY_URL]` or in `model_log/` as part of the archived release.

---

## Citation

If you use this repository, please cite:

```bibtex
@article{author_year_gesture,
  title   = {[Manuscript title]},
  author  = {[Author list]},
  journal = {[Journal name]},
  year    = {[Year]},
  doi     = {[Article DOI]}
}
```

Please also cite the archived code release:

```bibtex
@software{author_year_gesture_code,
  title  = {[Repository title]},
  author = {[Author list]},
  year   = {[Year]},
  doi    = {[Code DOI]},
  url    = {[Code archive URL]}
}
```

---

## License

Add a licence file before public release. For academic code intended for reuse, common options include MIT, BSD-3-Clause and Apache-2.0. Choose the licence that is compatible with institutional, funder and collaborator requirements.

---

## Contact

For questions about the code, data or trained models, contact:

```text
[Corresponding author name]
[Institution]
[Email address]
```

For issues related to the public repository, open an issue at `[REPOSITORY_ISSUES_URL]`.
