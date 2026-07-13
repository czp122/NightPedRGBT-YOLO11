# NightPedRGBT-YOLO11n (Ultralytics) + CBAM + LLVIP (RGBT 6-channel, dynamic pairing)

This project is an **Ultralytics official YOLO11** baseline (model YAML compatible with `yolo11n.yaml`), with a **minimal-intrusion patch**:

- 鉁?**CBAM attention**: added as a new module `CBAM` and a drop-in block `C3k2CBAM`.
- 鉁?**YAML insertion**: in `configs/yolo11n_llvip_rgbt_cbam.yaml` we replace `C3k2` with `C3k2CBAM` (Channel+Spatial attention) so CBAM is *actually in the graph*.
- 鉁?**LLVIP RGBT dynamic pairing**: **no fused images saved**. Visible+Infrared are loaded on-the-fly and concatenated into **6 channels (BGR + IR*3)**.
- 鉁?**One-click train** from PyCharm green run: all frequently-changed paths/params are in `.env`.

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
- Click PyCharm green 鈻讹笍 to run

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

