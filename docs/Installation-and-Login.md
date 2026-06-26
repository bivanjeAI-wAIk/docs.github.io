# Installation and Setup

> **Note:** wAIk does not include a login or authentication system. The application runs locally on your device — simply running it is enough.

---

## Prerequisites

Before installation, make sure you have the following available:

| Tool | Version | Note |
|---|---|---|
| Python | 3.12.8 | Exact version recommended |
| `uv` | latest | Package and virtual environment manager |
| Git | any | For cloning the repository |
| CUDA drivers | ≥ 11.8 | Optional; needed only for GPU-accelerated training |
| `just` | latest | Optional; simplifies common commands |

**Installing `uv`** (if you don't already have it):

```sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Installing `just`** (optional):

```sh
# Using uv (after installing uv)
uv tool install just

# Linux (Debian/Ubuntu)
sudo apt install just
```

---

## 1. Cloning and Installing Dependencies

Clone the repository and install all Python dependencies with a single command:

```sh
git clone <repo-url>
cd wAIk-projekt
uv sync
```

`uv sync` automatically creates a virtual environment (`.venv/`) and installs all dependencies from `pyproject.toml` into it. After this step, you don't need internet access for basic operation.

---

## 2. Preparing the Folder Structure

The application expects the following folders at the project root:

```
wAIk-projekt/
├── binaries/    ← raw .bin recordings from the IMU sensor
├── samples/     ← processed .npz samples
└── artifacts/   ← saved models, metrics, plots
```

**Option A — using `just`:**

```sh
just setup
```

**Option B — manual:**

```sh
mkdir binaries samples artifacts
```

The folders are empty on first install. You add the contents (recordings and samples) yourself, or import them using the `just pull-artifacts` command (see section 5).

---

## 3. Launching the GUI

```sh
uv run python app.py
```

Alternatively, if you have `just` installed:

```sh
just app
```

On startup, the main application window opens with the following tabs:

| Tab | Purpose |
|---|---|
| Dataset | Creating `.npz` samples from `.bin` recordings |
| Training | Training the CNN model for identity recognition |
| Artifact View | Reviewing saved models and metrics |
| Live Predict | Real-time prediction (requires the SPO service) |
| Predict | Prediction from existing `.npz` files |
| Sample View | Visualization of `.npz` samples |
| View Binaries | Visualization of raw `.bin` recordings |
| History | Reviewing and exporting prediction history |

---

## 4. Installing the SPO Service (Linux only, for real-time prediction)

This step is needed **only** if you want to run real-time predictions via a physical IMU sensor. This functionality is not supported on Windows and macOS.

The SPO service is a system daemon that listens on `127.0.0.1:5000` and forwards the binary data stream from the sensor to the application.

**1. Copy the service file:**

```sh
cp spo/spo.service ~/.config/systemd/user/spo.service
```

Instead of copying, you can create a symbolic link, which makes updates easier:

```sh
ln -s "$(pwd)/spo/spo.service" ~/.config/systemd/user/spo.service
```

**2. Enable and start the service:**

```sh
systemctl --user enable spo
systemctl --user start spo
```

**3. Check the status:**

```sh
systemctl --user status spo
```

The output must contain `active (running)`. After this, the service will start automatically on every login.

---

## 5. Downloading Pre-trained Artifacts (optional)

If you don't want to train the model yourself, you can download pre-trained weights and artifacts:

```sh
just pull-artifacts v0.1
```

The command downloads the saved models, XGBoost regressors, and metrics for the `v0.1` tag and stores them in the `artifacts/` folder. After this step, the *Predict* and *Artifact View* tabs are immediately usable.

---

## 6. Verifying the Installation

Run the test suite to verify the installation is correct:

```sh
uv run pytest -m "not slow and not local"
```

Tests tagged `slow` require a longer execution time (model training), while those tagged `local` require access to local data files or hardware. For a basic installation check, these are skipped.

A successful output looks like this:

```
============================= test session starts ==============================
...
===================== N passed in X.XXs =======================================
```

If errors occur, check:

- Whether the Python version is exactly 3.12.8 (`uv run python --version`)
- Whether all dependencies are installed (`uv sync`)
- Whether the `binaries/`, `samples/`, and `artifacts/` folders exist
