# Feature Documentation

This document describes eight key implemented features of the **wAIk** application.

---

## Feature 1: Binary Recording Parser

**Purpose of implementation:**
Raw IMU data captured by the SPO hardware is stored in binary form (`.bin`). To process it, a parser is needed that understands the transfer protocol — including sync markers, byte stuffing, and integrity checking.

**Implementation approach:**

- **File:** `src/waik/protocol.py`
- **Class:** `Parse`

The parser works as follows:
1. Searches for sync markers `0xFF 0xFF` in the binary data stream.
2. Performs **byte unstuffing** — unpacking special bytes inserted to separate packets.
3. Checks **CRC-16** for each packet to ensure data integrity.
4. Splits the stream into `Packet(id, timestamp, data)` objects.

Three sensor IDs are supported:
| ID | Sensor |
|----|--------|
| 1  | Gyroscope (angular velocity) |
| 2  | Accelerometer (acceleration) |
| 3  | Magnetometer |

**Usage:**
The parser is called internally via `SensorDataProcessor.load()` when loading a `.bin` file for processing. It is also directly accessible via the **View Binaries** tab, which loads and displays raw sensor data from a `.bin` file.

**Related issues:** SCRUM-6, SCRUM-12

**Known limitations:**
- Packets that fail CRC-16 validation are silently discarded. If a large portion of a recording is corrupted (e.g. due to a USB transfer error), the resulting signal will have gaps that may not be visible without inspecting the timestamp sequence.
- The parser expects `.bin` files produced specifically by the SPO service. Files from other IMU hardware or recording tools are not supported.

---

## Feature 2: Signal Processing

**Purpose of implementation:**
Raw int16 values from the sensors are not directly suitable for model training. Conversion to physical units, alignment of all sensor time series to a common sampling rate, and noise filtering are needed.

**Implementation approach:**

- **File:** `src/waik/preprocessing.py`
- **Class:** `SensorDataProcessor`

The processing pipeline proceeds through the following steps:

1. **`load()`** — loads and parses the `.bin` file using the `Parse` class.
2. **`process_all()`** — converts raw int16 values into physical units (e.g. m/s² for acceleration, °/s for gyroscope).
3. **`mag_resampling()`** — resamples all sensors to a uniform **100 Hz** rate, mean-centers the acceleration, and computes the normalized signal magnitude.
4. **`apply_runtime_lowpass()`** — applies a **Butterworth low-pass filter** with a 10 Hz cutoff frequency.

The output is saved in `.npz` format with the following keys:
- `acc_matrix`, `acc_mag_matrix`, `acc_fs`
- `gyro_matrix` and others

**Usage:**
The pipeline is triggered automatically when generating samples (**Dataset** tab) and during inference from a `.bin` file (**Live Predict** tab). The processed `.npz` files serve as input for the model and for visualization.

**Related issues:** SCRUM-8, SCRUM-9, SCRUM-10

**Known limitations:**
- The magnetometer (sensor ID 3) is parsed but not used in model training or inference. Its data is available in the `.npz` output but not currently passed to any downstream model.
- Resampling to 100 Hz assumes the original sensor rate is close to 100 Hz. Recordings captured at significantly different rates may produce aliasing or interpolation artifacts.
- Mean-centering of the acceleration removes the gravity component under the assumption of a roughly constant sensor orientation during walking. Recordings with large postural changes may not be correctly normalized.

---

## Feature 3: Sample Generation

**Purpose of implementation:**
Training models require fixed-length, labeled samples. Windows of a specific length must be cut from long gait recordings and saved with the appropriate metadata (identity, age, sex).

**Implementation approach:**

- **Files:** `src/waik/prepare_samples.py`, `src/waik/model/prepare_samples.py`

Procedure:
1. Reads `annotations.yaml` — a list of `.bin` recordings with user labels, time segments, and metadata.
2. Reads `dataset.yaml` — general dataset settings.
3. For each segment, cuts windows of **5 seconds** in length with a step of **2.5 seconds** (50% overlap).
4. Saves each window as a `.npz` file.

Naming convention:
```
{prefix}{index}_l{label}_{position}_{age}_{gender}.npz
```

**Usage:**
- **GUI:** **Dataset** tab → **Generate Samples** button
- **CLI:** `just samples`

**Related issues:** SCRUM-34, SCRUM-56

**Known limitations:**
- Windows shorter than 5 seconds (e.g. at the very end of a recording segment) are discarded. Short recordings may therefore produce fewer samples than expected.
- Sample generation does not perform data augmentation. All samples are derived directly from the original recordings; the dataset size is limited by the number and length of available `.bin` files.
- Re-running sample generation overwrites existing `.npz` files in the output folder if they share the same name. Changing the `prefix` in `dataset.yaml` between runs is recommended to avoid accidental overwriting.

---

## Feature 4: CNN Model for Identity Recognition

**Purpose of implementation:**
The goal is to recognize a person based on a gait sample. The convolutional neural network learns to distinguish between individuals by analyzing STFT spectrograms of the IMU signals.

**Implementation approach:**

- **File:** `src/waik/model/model.py`
- **Class:** `GaitIdentityModel`

Model architecture:

| Component | Description |
|------------|------|
| **Backbone** | EfficientNet-lite0 (via the `timm` library) |
| **Input** | 6-channel 224×224 image: 3-axis acceleration STFT + 3-axis gyroscope STFT |
| **Embedding head** | `Linear(num_features→512)` → `ReLU` → `Dropout` → `Linear(512→256)` → `ReLU` → `Dropout`, followed by L2 normalization of the resulting embedding |
| **Classifier** | `Linear(256→num_identities)` |
| **Auxiliary heads** *(optional)* | Prediction of age, height, weight, and sex (enabled with `use_auxiliary_heads=True`) |

Training is implemented in `src/waik/model/train_identity.py`.

**Usage:**
- **GUI:** **Training** tab → **Train Identity** button
- The output is `cnn_identity.pth` in the `artifacts/` folder.

**Related issues:** SCRUM-22, SCRUM-38, SCRUM-48, SCRUM-60

**Known limitations:**
- The model is a closed-set classifier: it can only predict identities present in the training set. A person not in the training set will be assigned to the closest known identity with low confidence, not rejected outright — the `one_of_us` flag (threshold 0.80) is the only indicator of an unknown subject.
- Recognition accuracy degrades if gait conditions differ significantly between training and inference (e.g. different footwear, walking speed, or sensor placement position).
- The minimum useful recording length for inference is approximately 7.5 seconds (enough for 2 overlapping windows). Shorter recordings produce a single-window prediction with no voting benefit.

---

## Feature 5: XGBoost Models for Biometric Attributes

**Purpose of implementation:**
In addition to identity, the application aims to predict a person's biometric attributes (age, height, weight, sex). Since the CNN embeddings already contain biometrically relevant information, the XGBoost models are trained on these vector representations.

**Implementation approach:**

- **Files:** `src/waik/model/train_age.py`, `src/waik/model/train_attributes.py`, `src/waik/model/xgboost_pipeline.py`

| Model | Type | Input | Output | Evaluation |
|-------|-----|------|-------|------------|
| **Age** | `XGBRegressor` | 256-dim CNN embedding | years | MAE, RMSE |
| **Height** | `XGBRegressor` | 310-dim (256-dim embedding + 54-dim band energy from STFT) | cm | MAE, RMSE |
| **Weight** | `XGBRegressor` | 310-dim | kg | MAE, RMSE |
| **Sex** | `XGBClassifier` | 310-dim | binary | accuracy |

Saved artifacts: `xgb_age.json`, `xgb_height.json`, `xgb_weight.json`, `xgb_sex.json`

**Usage:**
- **GUI:** **Training** tab → **Train Age**, **Train Attributes**, or **Train All** buttons
- `Train All` sequentially runs training of the CNN model and all XGBoost models.

**Related issues:** SCRUM-24, SCRUM-37, SCRUM-50, SCRUM-59

**Known limitations:**
- The XGBoost models depend on CNN embeddings, so they must be retrained whenever `cnn_identity.pth` is updated. Using XGBoost models trained on embeddings from a different CNN version will produce unreliable results.
- Prediction accuracy for age, height, and weight is limited by the diversity of the training dataset. The local dataset (3 subjects) is insufficient for generalizable attribute prediction; the Kaggle dataset (93 subjects) is recommended.
- The sex classifier outputs a binary prediction and does not support non-binary labels.

---

## Feature 6: Window-Voting Inference

**Purpose of implementation:**
A single 5-second window may not be sufficient for a reliable prediction, since gait is not always uniform. Aggregating predictions across all overlapping windows of the entire recording reduces the impact of local anomalies and increases reliability.

**Implementation approach:**

- **File:** `src/waik/model/inference.py`
- **Class:** `InferenceService`
- **Method:** `predict_npz()`

Inference procedure:

1. Cuts overlapping windows of **5 seconds** in length with a step of **2.5 seconds** (250 samples at 100 Hz) from the entire recording.
2. Computes a **6-channel STFT image** (224×224) for each window.
3. Runs the CNN for each window and collects **softmax probabilities**.
4. **Averages the probabilities** across all windows → final identity prediction.
5. Also averages the 256-dimensional embeddings → input for the age XGBoost model.
6. Computes the 310-dimensional extended features (embedding + band energies) → input for the height, weight, and sex models.

The confidence threshold (default **0.80**) determines the value of the `one_of_us` flag:
- Value above the threshold → the person is recognized as "one of us"
- Value below the threshold → the person is not in the training set

**Usage:**
- **GUI:** **Predict** tab (for existing `.npz` files) and **Live Predict** tab (for live recordings)

**Related issues:** SCRUM-43

**Known limitations:**
- The voting strategy averages softmax probabilities uniformly across all windows. Windows containing non-walking segments (e.g. the subject standing still at the start or end of a recording) are not filtered out and may dilute the prediction.
- The confidence threshold of 0.80 is a fixed default and is not automatically calibrated to the training dataset. For datasets with many similar-looking identities, a higher threshold may be needed to reduce false positives.

---

## Feature 7: SPO Communication Service

**Purpose of implementation:**
To capture live IMU data, the application communicates with a specialized hardware-software service (SPO) that directly drives the sensor device. Communication occurs via a local TCP socket, which provides clean separation between the GUI application and the driver layer.

**Implementation approach:**

- **Files:** `src/waik/spo/client.py`, `spo/communication.py`, `spo/spo.service`

Communication flow:

1. The TCP client connects to **localhost:5000**.
2. It sends the command `STREAM|N`, where `N` is the number of seconds to record.
3. The SPO service records IMU data and saves it to a `.bin` file in the `binaries/` folder.
4. The service responds with:
   ```
   Stream captured to <filename>
   ```
5. The `capture_stream_binary()` function in `live_capture.py` then sequentially calls:
   - `create_live_prediction_sample()` — converts `.bin` to `.npz`
   - `predict_sample()` — runs inference

The SPO service runs as a **Linux systemd user service** (defined in `spo/spo.service`).

**Usage:**
- Starting the service: `systemctl --user start spo`
- Checking status: `systemctl --user status spo`
- **GUI:** **Live Predict** tab → **Capture + Predict** button

**Related issues:** SCRUM-31, SCRUM-44

**Known limitations:**
- The SPO service is Linux-only due to its systemd dependency. Windows and macOS users cannot use the Live Predict feature.
- If the TCP connection times out (e.g. the service is not running or port 5000 is blocked), the GUI displays a connection error. There is no automatic retry; the user must manually restart the service and click **Capture + Predict** again.
- Only one client can connect to the SPO service at a time. Running multiple application instances simultaneously will cause the second connection to fail.

---

## Feature 8: Prediction History

**Purpose of implementation:**
During a session, the user makes multiple predictions from different sources (manually loaded samples, live recordings). A historical record together with visualization allows comparing results, spotting trends, and exporting data for further analysis.

**Implementation approach:**

- **File:** `src/waik/gui/tabs/history_tab.py`
- **Class:** `HistoryTab`

Features:

- **Prediction table** with the following columns:
  `#` | `Time` | `Source` | `Identity` | `Age` | `Confidence`

- **Live scatter plot** with two Y axes:
  - Left axis: predicted age — green dots = recognized person, red dots = unrecognized
  - Right axis: confidence in % (orange dashed line)

- **Statistics row:** total number of predictions, number of recognized (Known) and unrecognized (Unknown)

- **Export CSV button:** saves the entire history to a CSV file chosen by the user

Each tab (**Predict** and **Live Predict**) calls `add_prediction(result: PredictionResult, source: str)`, which adds the prediction to the list and refreshes the table and chart.

**Usage:**
- The **History** tab is the last tab in the main application window.
- It is accessible at any time during the session; it updates automatically with each new prediction.
- Data export: **Export CSV** button → choose a path to save the `.csv` file.

**Related issues:** SCRUM-57, SCRUM-58

**Known limitations:**
- **History is session-only and is not persisted to disk.** Closing the application clears all history. Use the **Export CSV** button before closing if the data needs to be retained.
- The scatter plot does not update in real time during a long inference run; it refreshes only after each prediction is fully completed and `add_prediction()` is called.
- The CSV export includes only the columns shown in the table (`#`, `Time`, `Source`, `Identity`, `Age`, `Confidence`). Additional prediction details (height, weight, sex, `one_of_us` flag) are not included in the export.
