# Use Cases

This document describes the five main use cases of the **wAIk** application — a desktop tool for capturing IMU sensor data, training models, and recognizing identity and biometric attributes based on gait.

> **Note:** wAIk does not have a login or authentication system. The application is available immediately upon launching it with `uv run python app.py` (or `just app`). All use cases begin from the main application window.

---

## Use Case 1: Creating a Dataset

**What the user wants to do:**
The user wants to create `.npz` samples from raw binary recordings (`.bin` files) to be used for training models.

**Execution steps:**

1. Open the **Dataset** tab in the application.
2. In the path field, set the location of the `annotations.yaml` file (contains a list of `.bin` recordings with user labels, time segments, and metadata).
3. Set the path to the `dataset.yaml` file (contains general dataset settings).
4. Set the output folder for samples, e.g. `samples/`.
5. Click the **Generate Samples** button.
6. Wait for a message in the log of the form:
   ```
   Generated N samples.
   ```

**Result:**
`.npz` files are created in the `samples/` folder, named according to the convention:
```
{prefix}{index}_l{label}_{position}_{age}_{gender}.npz
```
These files are ready for the next step — model training.

**Troubleshooting:**
- If the log shows `0 samples generated`, verify that the `.bin` files listed in `annotations.yaml` exist and that the time segments are within the recording duration.
- If a file cannot be parsed, check that the `.bin` file was produced by the SPO service and is not corrupted (incomplete transfer).

---

## Use Case 2: Training a Model

**What the user wants to do:**
The user wants to train a CNN model for identity recognition and XGBoost models for predicting biometric attributes (age, height, weight, sex).

**Execution steps:**

1. Open the **Training** tab in the application.
2. Set the path to the samples folder (e.g. `samples/`).
3. Set the path to the output folder for artifacts (e.g. `artifacts/`).
4. Configure the hyperparameters:
   - Number of epochs (`epochs`)
   - Batch size (`batch size`)
   - Validation split (`validation split`)
   - Early stopping (`early stopping`)
5. Click **Train Identity** to train the CNN identity recognition model, or **Train All** to sequentially train all models (identity + age + attributes).
6. Monitor progress via the progress bar and log. Log message format:
   ```
   epoch=3/40 batch=120/246 val_loss=0.612 val_acc=0.834
   ```
7. Training can be **paused and resumed** with the Pause / Resume buttons.
8. After completion, a dialog may appear asking whether to save a checkpoint.

**Result:**
The following files are saved in the `artifacts/` folder:
- `cnn_identity.pth` — the trained CNN model
- `xgb_age.json`, `xgb_height.json`, `xgb_weight.json`, `xgb_sex.json` — XGBoost models
- `identity_labels.json`, `thresholds.json`, `train_config.json`, `training_metrics.json`

Training charts can be viewed in the **Artifact View** tab.

**Troubleshooting:**
- If `val_acc` stays near `1/N` (where N is the number of identities), the model is not learning — check that samples from multiple identities are present and that the samples folder path is correct.
- GPU training requires CUDA drivers ≥ 11.8; if CUDA is unavailable, training falls back to CPU automatically (slower).

---

## Use Case 3: Analyzing a Recording

**What the user wants to do:**
The user wants to identify a person and obtain a prediction of their biometric attributes based on an existing `.npz` gait recording.

**Execution steps:**

1. Open the **Predict** tab in the application.
2. Use the file selection button to choose a `.npz` gait sample.
3. Set the path to the `artifacts/` folder containing the trained models.
4. Click the **Predict** button.
5. Wait for the model to process the sample (multi-window inference with voting).
6. The following values are displayed:
   - **Age** — predicted age
   - **Height** — predicted height
   - **Weight** — predicted weight
   - **Sex** — predicted sex
   - **One of us: yes / no** — whether the person is in the training set
   - **Identity** — name of the recognized person (e.g. `miha`, `leon`, `luka`)
   - **Confidence** — prediction confidence as a percentage
7. The prediction is automatically added to the **History** tab.

**Result:**
The predicted biometric attributes and the person's identity are displayed together with a confidence value. A confidence of **≥ 80%** means the person is classified as *One of us: yes* (a known identity from the training set). Below 80%, the person is considered unknown — the model has seen a gait it does not recognize.

**Troubleshooting:**
- If `artifacts/` is empty, either run training first or download pre-trained artifacts with `just pull-artifacts v0.1`.
- If the recording is very short (< 5 seconds of usable walking), only one window will be evaluated, which may reduce accuracy.

---

## Use Case 4: Live Prediction

**What the user wants to do:**
The user wants to identify a walking person in real time, using the SPO hardware and a live IMU data stream.

> **Prerequisite:** This use case requires Linux and a connected IMU sensor unit. The SPO service must be installed (see [Installation and Setup](Installation-and-Login)).

**Execution steps:**

1. Make sure the SPO service is running. On Linux, check the status with the command:
   ```bash
   systemctl --user status spo
   ```
   The output must contain `active (running)`.
2. Open the **Live Predict** tab in the application.
3. Set the recording duration (e.g. `10` seconds).
4. Set the path to the `artifacts/` folder with the trained models.
5. Click the **Capture + Predict** button.
6. The application sends the command `STREAM|10` to the SPO service via a TCP connection (localhost:5000).
7. The SPO service records IMU data and saves it to a `.bin` file in the `binaries/` folder, then responds with:
   ```
   Stream captured to <filename>
   ```
8. The application automatically converts the `.bin` to `.npz`, runs inference, and displays the results:
   - **One of us** — whether the person is recognized
   - **Who** — name of the identity
   - **Confidence** — confidence as a percentage
   - **Age**, **Height**, **Weight**, **Sex** — biometric attributes
   - Paths to the saved `.bin` and `.npz` files

**Result:**
The result of the real-time gait identification is displayed and saved. The binary file and the sample (`.npz`) are available for later analysis in the **Sample View** or **Predict** tabs. The prediction is added to the **History** tab.

**Troubleshooting:**
- If the button produces a connection error, verify the SPO service is running (`systemctl --user status spo`) and that nothing else is listening on port 5000.
- If the result shows *One of us: no* for a person who should be recognized, try a longer recording duration (15–20 seconds) to provide more windows for voting.

---

## Use Case 5: Signal Visualization

**What the user wants to do:**
The user wants to review the raw acceleration and angular velocity signal from a gait recording and assess its quality before using it for training or analysis.

**Execution steps:**

1. Open the **Sample View** tab in the application.
2. Use the file selection button to choose a `.npz` file.
3. Set the **start time** (in seconds) — where in the recording to begin the display.
4. Set the **duration** (in seconds) — how much of the recording to display.
5. Click the **Load Sample** button.
6. A 2×2 grid of plots is displayed:
   - **Top left:** time-domain acceleration (accelerometer) signal on the X, Y, Z axes
   - **Top right:** STFT spectrogram of acceleration
   - **Bottom left:** time-domain angular velocity (gyroscope) signal on the X, Y, Z axes
   - **Bottom right:** STFT spectrogram of angular velocity

**Result:**
A visual representation of the gait signal is shown. Use the plots to assess recording quality:

| What you see | Interpretation |
|---|---|
| Regular oscillations at ~1–2 Hz in the time-domain plots | Normal walking cadence — good quality recording |
| Flat or near-zero signal on one or more axes | Sensor may have been stationary, disconnected, or clipped |
| Sudden large spikes | Impact artifacts (e.g. the person stumbled or the sensor shifted) — consider excluding this time segment in `annotations.yaml` |
| Energy concentrated in 0–10 Hz bands in the spectrograms | Expected for walking; the Butterworth low-pass filter cutoff is 10 Hz |
| Broadband noise across all frequencies in the spectrogram | High-frequency noise; verify the low-pass filter was applied during preprocessing |

For raw `.bin` files (before preprocessing), use the **View Binaries** tab instead, which shows the unparsed sensor packets directly.
