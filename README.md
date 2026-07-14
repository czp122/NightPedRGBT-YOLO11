# NightPedRGBT-YOLO11n (Ultralytics) + CBAM + LLVIP (RGBT 6-channel, dynamic pairing)

Current application version: **v1.1.0**

This project is an **Ultralytics official YOLO11** baseline (model YAML compatible with `yolo11n.yaml`), with a **minimal-intrusion patch**:

- **CBAM attention**: added as a new module `CBAM` and a drop-in block `C3k2CBAM`.
- **YAML insertion**: in `configs/yolo11n_llvip_rgbt_cbam.yaml` we replace `C3k2` with `C3k2CBAM` (Channel+Spatial attention) so CBAM is *actually in the graph*.
- **LLVIP RGBT dynamic pairing**: **no fused images saved**. Visible+Infrared are loaded on-the-fly and concatenated into **6 channels (BGR + IR*3)**.
- **One-click train** from PyCharm: all frequently-changed paths/params are in `.env`.

## 1. Dataset layout (expected)

```
LLVIP/
  visible/
    train/images/*.jpg
    train/labels/*.txt
    test/images/*.jpg
    test/labels/*.txt
  infrared/
    train/images/*.jpg
    test/images/*.jpg
```

## 2. Quick start (PyCharm)

1) Create a python environment (recommend Python 3.9+)

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Copy and edit env file:

- Copy `.env.example` -> `.env`
- Edit `LLVIP_ROOT` to your dataset path
- (Windows) keep `WORKERS=0` to avoid multiprocessing issues

4) Run training:

- Open `scripts/train.py`
- Click the PyCharm green Run button

The script will:
- auto-generate `runs/llvip_rgbt_data.yaml`
- patch Ultralytics at runtime to load 6ch RGBT
- run **official Ultralytics trainer** (prints P/R/F1/mAP)


## 2.5. Switchable experiment configs

You can now choose an experiment profile without editing the root `.env` each time:

```bash
python scripts/train.py --env configs/experiments/exp1_ft800.env
python scripts/train.py --env configs/experiments/exp2_ft960.env
python scripts/train.py --env configs/experiments/exp3_yolo11s_800.env
```

Profiles included in `configs/experiments/`:

- `exp1_ft800.env`: continue from the current best YOLO11n checkpoint at `imgsz=800`
- `exp2_ft960.env`: continue from the current best YOLO11n checkpoint at `imgsz=960`
- `exp3_yolo11s_800.env`: train a larger YOLO11s RGBT+CBAM model from scratch

If you move the project or the checkpoint, update the absolute `WEIGHTS=` path inside the experiment env file.
## 3. Verify CBAM is actually loaded

Run:

```bash
python scripts/check_cbam.py
```

It will print how many `C3k2CBAM` blocks and `CBAM` modules are present.

## 4. How the dynamic pairing works (no extra disk)

The runtime patch in `rgbt/patch_ultralytics.py` monkey-patches Ultralytics:

- `BaseDataset.load_image()`
  - reads visible image
  - maps path `.../visible/... -> .../infrared/...`
  - reads infrared image
  - repeats IR to 3 channels
  - concatenates to 6 channels

We also patch `RandomHSV` so HSV augmentation only affects the **visible 3 channels** when the image is 6ch.

## 5. Where CBAM is inserted

- **New module**: `ultralytics/nn/modules/conv.py` -> `ChannelAttention`, `SpatialAttention`, `CBAM`
- **New block**: `ultralytics/nn/modules/block.py` -> `C3k2CBAM` (wraps C3k2 and applies CBAM)
- **YAML**: `configs/yolo11n_llvip_rgbt_cbam.yaml` replaces `C3k2` with `C3k2CBAM` in Backbone+Neck.

## Notes

- 6-channel input **cannot directly use** official pretrained weights (3-channel). Training uses `pretrained=False`.
- If your infrared images are single-channel, the patch repeats to 3ch automatically.

## 6. Runtime acceleration

The GUI device selector defaults to `auto`:

- NVIDIA CUDA is available: use GPU 0, FP16 inference, cuDNN benchmark and TF32 automatically.
- CUDA is unavailable: use CPU automatically, with bounded PyTorch/OpenCV thread counts to avoid thread contention.
- CUDA inference fails or runs out of memory: clear the CUDA cache, reset the predictor and retry on CPU.
- The status bar displays the actual device, inference latency and runtime FPS.

Run the source version:

```powershell
.\.venv\Scripts\python.exe gui\app.py
```

The source entry points set `MKL_THREADING_LAYER=SEQUENTIAL` before importing PyTorch. This prevents Conda MKL and
PyTorch from initializing different Intel OpenMP runtimes during ByteTrack inference. Do not use the unsafe
`KMP_DUPLICATE_LIB_OK=TRUE` workaround.

When running from PyCharm with a Conda interpreter, also set `MKL_THREADING_LAYER=SEQUENTIAL` in the app Run/Debug
Configuration so the value exists before PyCharm starts Python. A shared `NightPedestrianDetection` run configuration
is included in `.run/NightPedestrianDetection.run.xml`; select it after opening the project.

On a computer with a supported NVIDIA GPU and driver, install the CUDA-enabled PyTorch build that matches this project:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_cuda_acceleration.ps1
```

Then start the GUI and keep `device=auto`. A self-contained EXE only includes the PyTorch runtime present when it was built. To create a GPU-capable EXE, run the CUDA installation script first and then run `build_exe.ps1` on the NVIDIA computer.

For 6-channel video on CPU, realtime IR preprocessing uses a reduced NLM search area while preserving NLM denoising and CLAHE. Full-resolution preprocessing remains unchanged for dataset images and training.

## 7. v1.1.0 application features

- Separate `Realtime Detection` and `Dataset Evaluation` control tabs.
- ByteTrack pedestrian IDs, current count, unique count and alert-entry count.
- Mouse-drawn alert ROI with audible warning, snapshot, JSON metadata and pre/post-event MP4 recording.
- RGB/IR frame offset, manual X/Y translation and synchronized dual-camera `grab/retrieve`.
- Quality, Balanced and Smooth performance modes. Smooth mode performs alternate RGBT inference and reuses tracked boxes without skipping source frames.
- CSV/JSON detection record export with timestamps, IDs, confidence, boxes, device and FPS.
- Runtime diagnostics for CPU, GPU, CUDA, PyTorch, OpenCV, model channels and memory usage.
- Persistent settings under `runs/gui_config/settings.json` in source mode or `%LOCALAPPDATA%/NightPedRGBT` in the packaged application.

Detailed Chinese instructions are available in `docs/v1.1.0使用说明.md`.

