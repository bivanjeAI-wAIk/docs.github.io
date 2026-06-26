# wAIk — Project Specifications

## 1. Project Purpose

**wAIk** is a desktop application for gait analysis based on deep learning. The system captures accelerometer and gyroscope data from a wearable IMU (inertial measurement unit) and uses it to recognize an individual by their gait pattern or predict biometric characteristics (age, height, weight, sex).

### Target User Groups

| Group | Technical Level | Needs |
|---|---|---|
| Researchers | Intermediate — comfortable with file paths and YAML configuration | Collecting and annotating gait recordings, training and comparing models, exporting results |
| Developers | Advanced — familiar with Python, ML pipelines, and CLI tools | Dataset integration, model architecture customization, pipeline automation |

### User Needs

- Easy capture and organization of raw IMU recordings (`.bin` files)
- Automatic conversion of recordings into samples suitable for machine learning
- Training models for identity recognition and biometric attribute prediction directly from the application
- Visualization of signals and training results without needing external tools
- Real-time prediction via physical hardware or from existing `.npz` files
- Tracking prediction history and exporting results

---

## 2. Solution Overview

wAIk covers the entire pipeline — from raw sensor data to the final prediction — within a single desktop application.

```
Sensor (IMU)
    │
    ▼
SPO Service (TCP, localhost:5000)
    │  binary data stream
    ▼
Binary Parser (protocol.py)
    │  parsing, CRC-16, byte stuffing
    ▼
Signal Processing (preprocessing.py)
    │  resampling to 100 Hz, low-pass filter, STFT
    ▼
Samples (.npz)
    ├──▶ CNN Model — GaitIdentityModel (EfficientNet-lite0)
    │        │  multi-window voting prediction
    │        ▼
    │    Person Identity
    │
    └──▶ XGBoost Models
             │  regressors / classifier
             ▼
         Age · Height · Weight · Sex
```

The graphical interface (PySide6) allows management of all pipeline steps without using the command line.

### Assumptions and Dependencies

| Assumption / Dependency | Details |
|---|---|
| SPO hardware required for live prediction | The Live Predict feature requires a physical IMU sensor unit connected via the SPO service. Offline features (Predict, Sample View, Training) work without hardware. |
| Linux required for SPO service | The systemd-based SPO service only runs on Linux. Windows and macOS users can use all offline features but cannot perform live recordings. |
| Pre-trained artifacts optional | The Predict and Artifact View tabs require trained models. Users can either train their own or download the `v0.1` pre-trained artifacts via `just pull-artifacts v0.1`. |
| Third-party libraries | The project depends on `timm` (EfficientNet-lite0 backbone), `xgboost`, `PySide6`, `scipy`, `numpy`, and `torch`. All are managed via `uv` from `pyproject.toml`. |
| Python version | Exactly Python 3.12.8 is required. Other versions may work but are not tested. |
| Dataset for training | Training the CNN requires labeled `.bin` recordings described in `annotations.yaml`. The application supports local recordings and the Kaggle gait dataset (93 subjects). |

---

## 3. Functional Requirements

### 3.1 Dataset Creation

- Reading and parsing raw binary recordings (`.bin`) based on labels in `annotations.yaml`
- Splitting recordings into time segments and exporting as `.npz` samples
- Support for local recordings (miha, leon, luka) and the online Kaggle dataset (93 people)
- Path and parameter configuration in `dataset.yaml`

**Acceptance criteria:**
- Given a valid `annotations.yaml` and `dataset.yaml`, clicking **Generate Samples** produces `.npz` files in the output folder named according to the `{prefix}{index}_l{label}_{position}_{age}_{gender}.npz` convention.
- The log displays `Generated N samples.` upon completion, where N matches the expected number of windowed segments.

### 3.2 Identity Recognition Model Training

- Training a CNN model (`GaitIdentityModel`) with an EfficientNet-lite0 backbone on STFT spectrograms
- Hyperparameter configuration (number of epochs, batch size, learning rate) via the interface
- Saving artifacts (model weights, metrics, plots) to the `artifacts/` folder

**Acceptance criteria:**
- Clicking **Train Identity** starts training and displays per-epoch progress in the format `epoch=N/M batch=... val_loss=... val_acc=...`.
- Upon completion, `cnn_identity.pth`, `identity_labels.json`, `thresholds.json`, `train_config.json`, and `training_metrics.json` are present in `artifacts/`.
- Training can be paused and resumed without loss of progress.

### 3.3 Training Models for Biometric Attribute Prediction

- Training XGBoost regressors for age, height, and weight
- Training an XGBoost classifier for sex
- Joint attribute training via `train_attributes.py`

**Acceptance criteria:**
- Clicking **Train All** produces `xgb_age.json`, `xgb_height.json`, `xgb_weight.json`, and `xgb_sex.json` in `artifacts/`.
- MAE and RMSE metrics for regression models are written to the training log.

### 3.4 Offline Prediction

- Loading existing `.npz` samples and running predictions with a trained model
- Displaying predicted identity and biometric attributes
- Multi-window voting strategy (`InferenceService`) for more robust results

**Acceptance criteria:**
- Given a valid `.npz` file and a populated `artifacts/` folder, clicking **Predict** displays identity, confidence, age, height, weight, and sex within 10 seconds.
- A person with softmax confidence ≥ 0.80 is classified as *One of us: yes*; below this threshold as *no*.
- Each prediction is automatically appended to the **History** tab.

### 3.5 Real-Time (Live) Prediction

- Connection to the SPO service via TCP socket (`localhost:5000`)
- Streaming and parsing of the live binary stream during walking
- Real-time display of identity and biometric attribute predictions

**Acceptance criteria:**
- With the SPO service running, clicking **Capture + Predict** initiates a recording of the configured duration, then displays identity and attribute predictions without manual intervention.
- The captured `.bin` and converted `.npz` files are saved in `binaries/` and `samples/` respectively.

### 3.6 Signal Visualization

- Display of signals in the time domain for `.npz` and `.bin` files
- Display of STFT spectrograms
- *Sample View* and *View Binaries* tabs in the interface

**Acceptance criteria:**
- Loading a `.npz` file in **Sample View** displays a 2×2 grid: time-domain acceleration, acceleration STFT spectrogram, time-domain gyroscope, and gyroscope STFT spectrogram.
- The user can set an arbitrary start time and duration; the plots update accordingly.

### 3.7 Artifact Review

- Browsing saved models, loss/accuracy curves, and plots in the *Artifact View* tab

**Acceptance criteria:**
- With a populated `artifacts/` folder, the **Artifact View** tab lists available model versions and displays loss and accuracy curves for the selected model.

### 3.8 Prediction History

- Logging all performed predictions with labels (time, identity, attributes)
- Exporting history to CSV format from the *History* tab

**Acceptance criteria:**
- All predictions made during a session appear in the **History** tab table with correct time, source, identity, age, and confidence values.
- Clicking **Export CSV** opens a file dialog and saves a valid CSV file containing all history rows.
- Note: history is session-only and does not persist after the application is closed.

---

## 4. Non-Functional Requirements

### 4.1 Performance

| Metric | Requirement |
|---|---|
| Offline inference latency | A single `.npz` prediction (all windows) shall complete within **10 seconds** on a CPU-only machine with ≥ 8 GB RAM. |
| Live capture overhead | After the SPO service returns the `.bin` file, the `.npz` conversion and inference shall complete within **15 seconds**. |
| Training throughput | CNN training shall process at least **50 batches per minute** on a CUDA-capable GPU. |

### 4.2 Usability

- All pipeline steps (dataset creation, training, prediction, visualization) shall be accessible through the GUI without requiring use of the command line.
- Error messages shall be displayed in the application log in plain language, identifying the failed step and suggesting a corrective action.
- The application shall start and display the main window within **5 seconds** on supported hardware.

### 4.3 Reliability

- The binary parser shall discard packets that fail CRC-16 validation and continue processing the remainder of the stream without crashing.
- If the TCP connection to the SPO service fails, the **Live Predict** tab shall display an informative error and allow the user to retry without restarting the application.

### 4.4 Portability

- All offline features (Dataset, Training, Predict, Sample View, View Binaries, Artifact View, History) shall function on Windows 10+, macOS 12+, and Ubuntu 22.04+.
- The SPO service and **Live Predict** feature are Linux-only (systemd dependency).

### 4.5 Maintainability

- The project shall use `uv` for dependency management; adding or updating a dependency requires only editing `pyproject.toml` and running `uv sync`.
- The test suite (`pytest`) shall cover core processing and inference logic; running `uv run pytest -m "not slow and not local"` shall complete without errors on a clean install.

---

## 5. System Requirements

| Requirement | Details |
|---|---|
| Python | 3.12.8 |
| Package manager | `uv` |
| RAM | ≥ 8 GB (recommended) |
| GPU | CUDA-capable graphics card (optional; speeds up training) |
| Operating system | Windows, macOS, or Linux for offline functionality; **Linux** for the SPO service (real-time prediction) |
| Hardware | IMU sensor unit (required only for capturing live data) |

---

## 6. Glossary

| Term | Definition |
|---|---|
| **IMU** | Inertial Measurement Unit — a sensor device that measures acceleration (accelerometer) and angular velocity (gyroscope), and optionally magnetic field (magnetometer). In wAIk, the IMU is worn on the body during walking. |
| **SPO service** | A Linux systemd user service that communicates directly with the IMU hardware and forwards binary sensor data to the application via a TCP socket on `localhost:5000`. |
| **`.bin` file** | A raw binary recording produced by the SPO service, containing timestamped sensor packets with sync markers, byte stuffing, and CRC-16 integrity checks. |
| **`.npz` file** | A NumPy archive file containing processed sensor arrays (acceleration matrix, gyroscope matrix, sampling frequency, etc.) ready for model input or visualization. |
| **STFT** | Short-Time Fourier Transform — a method for computing the frequency content of a signal over time. In wAIk, STFT spectrograms of accelerometer and gyroscope signals are used as 2D image inputs to the CNN. |
| **CNN** | Convolutional Neural Network — a type of deep learning model suited for image-like input. wAIk uses `GaitIdentityModel` with an EfficientNet-lite0 backbone to classify gait identity from STFT spectrograms. |
| **EfficientNet-lite0** | A lightweight convolutional neural network architecture from the EfficientNet family, loaded via the `timm` library. Used as the feature-extraction backbone of `GaitIdentityModel`. |
| **XGBoost** | Extreme Gradient Boosting — a gradient-boosted decision tree library. wAIk uses XGBoost regressors (age, height, weight) and a classifier (sex) trained on CNN embeddings. |
| **Embedding** | A compact numerical vector (256-dimensional in wAIk) produced by the CNN's embedding head, encoding biometrically relevant features of a gait window. |
| **CRC-16** | Cyclic Redundancy Check (16-bit) — an error-detection algorithm used to verify the integrity of each sensor packet in the `.bin` stream. |
| **Byte stuffing** | A framing technique used in the binary protocol to distinguish data bytes from packet delimiters (sync markers `0xFF 0xFF`), preventing false packet boundaries. |
| **Window / sliding window** | A fixed-length time segment (5 seconds, 500 samples at 100 Hz) cut from a longer recording. Multiple overlapping windows are extracted per recording with a 2.5-second step (50% overlap). |
| **Window voting** | An inference strategy where softmax predictions from all windows of a recording are averaged to produce a final, more robust identity prediction. |
| **Confidence** | The averaged softmax probability for the predicted identity class, expressed as a percentage. Values ≥ 80% classify the person as *One of us*. |
| **One of us** | A binary flag indicating whether the predicted person is present in the training set (`yes` if confidence ≥ 0.80, `no` otherwise). |
| **Artifacts** | Files produced during or after training, stored in the `artifacts/` folder: model weights (`.pth`), XGBoost models (`.json`), metrics, and plots. |
| **`annotations.yaml`** | A configuration file listing `.bin` recordings with associated metadata: subject identity, recording time segments, sensor placement position, age, and sex. |
| **`dataset.yaml`** | A configuration file specifying global dataset parameters such as window length, overlap, output path, and included subjects. |
| **`uv`** | A fast Python package and virtual environment manager used to install dependencies and run scripts (`uv sync`, `uv run`). |
| **`just`** | An optional command runner that provides short aliases for common project commands (`just app`, `just samples`, `just pull-artifacts`). |
| **Butterworth filter** | A type of signal filter with a maximally flat frequency response. wAIk applies a low-pass Butterworth filter (10 Hz cutoff) to reduce high-frequency noise in the sensor signals. |
| **100 Hz** | The uniform sampling rate to which all sensor streams are resampled during preprocessing. |
| **PySide6** | The official Python binding for the Qt 6 framework, used to build the wAIk graphical user interface. |
