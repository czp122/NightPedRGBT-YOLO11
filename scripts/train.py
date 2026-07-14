from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

DEFAULT_ENV = ROOT / ".env"

from rgbt_config import AppConfig
from rgbt.patch_ultralytics import apply_rgbt_llvip_patch, llvip_sanity_check


def write_llvip_data_yaml(cfg: AppConfig) -> str:
    out = (ROOT / cfg.data_yaml_out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "path": str(Path(cfg.llvip_root).resolve()),
        "train": cfg.train_split,
        "val": cfg.val_split,
        "test": cfg.test_split,
        "names": {0: "person"},
        "nc": 1,
        "channels": 6,
    }
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return str(out)


def _plot_metrics_csv(csv_path: Path) -> None:
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return

    epochs = list(range(1, len(rows) + 1))
    metric_keys = [
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]

    values = {}
    for key in metric_keys:
        series = []
        for row in rows:
            try:
                series.append(float(row.get(key, "") or 0.0))
            except Exception:
                series.append(0.0)
        if any(v != 0.0 for v in series):
            values[key] = series

    if not values:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150)
    loss_keys = [k for k in values if "loss" in k]
    metric_only_keys = [k for k in values if "loss" not in k]

    for key in loss_keys:
        axes[0].plot(epochs, values[key], label=key)
    axes[0].set_title("Training / Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    for key in metric_only_keys:
        axes[1].plot(epochs, values[key], label=key)
    axes[1].set_title("Precision / Recall / mAP")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(csv_path.with_name("metrics_curves.png"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LLVIP RGBT YOLO model.")
    parser.add_argument(
        "--env",
        default=None,
        help=f"Optional path to an experiment .env file. Defaults to {DEFAULT_ENV}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_path = Path(args.env).expanduser() if args.env else DEFAULT_ENV
    if not env_path.is_absolute():
        env_path = ROOT / env_path
    env_path = env_path.resolve()
    if not env_path.is_file():
        raise FileNotFoundError(
            f"Training config not found: {env_path}. Copy .env.example to .env and set LLVIP_ROOT."
        )
    cfg = AppConfig.from_env(str(env_path))
    if not cfg.llvip_root.strip():
        raise ValueError(f"LLVIP_ROOT is empty in training config: {env_path}")
    if not 32 <= cfg.imgsz <= 2048:
        raise ValueError(f"IMGSZ must be between 32 and 2048, got {cfg.imgsz}")
    if cfg.epochs < 1:
        raise ValueError(f"EPOCHS must be at least 1, got {cfg.epochs}")
    if cfg.batch == 0 or cfg.batch < -1:
        raise ValueError(f"BATCH must be -1 or a positive integer, got {cfg.batch}")

    print(f"[Config] Using env file: {env_path}")
    print(f"[Config] Run name: {cfg.name}")
    print(f"[Config] Model YAML: {cfg.model_yaml}")
    if cfg.weights:
        print(f"[Config] Fine-tuning from: {cfg.weights}")
    print(f"[Config] Optimizer: {cfg.optimizer}, lr0={cfg.lr0}")

    llvip_sanity_check(cfg.llvip_root)
    apply_rgbt_llvip_patch()
    data_yaml = write_llvip_data_yaml(cfg)

    from ultralytics import YOLO

    model_yaml_path = (ROOT / cfg.model_yaml).resolve()
    if not model_yaml_path.is_file():
        raise FileNotFoundError(f"Model YAML not found: {model_yaml_path}")
    model_yaml = str(model_yaml_path)
    weights = (cfg.weights or "").strip()
    use_finetune = bool(weights)
    weights_path = Path(weights).expanduser()
    if use_finetune and not weights_path.is_absolute():
        weights_path = (ROOT / weights_path).resolve()
    if use_finetune and not weights_path.is_file():
        raise FileNotFoundError(f"Fine-tuning weights not found: {weights_path}")
    model_source = str(weights_path) if use_finetune else model_yaml
    model = YOLO(model_source)
    project_path = Path(cfg.project).expanduser()
    if not project_path.is_absolute():
        project_path = ROOT / project_path
    project_path = project_path.resolve()

    results = model.train(
        data=data_yaml,
        imgsz=cfg.imgsz,
        epochs=cfg.epochs,
        batch=cfg.batch,
        workers=cfg.workers,
        device=cfg.device,
        lr0=cfg.lr0,
        weight_decay=cfg.weight_decay,
        optimizer=cfg.optimizer,
        amp=bool(cfg.amp),
        cache=bool(cfg.cache),
        seed=cfg.seed,
        pretrained=use_finetune,
        patience=cfg.patience,
        plots=bool(cfg.plots),
        val=True,
        save=True,
        project=str(project_path),
        name=cfg.name,
        verbose=False,
    )

    save_dir = Path(getattr(results, "save_dir", project_path / cfg.name))
    csv_path = save_dir / "results.csv"
    _plot_metrics_csv(csv_path)

    try:
        metrics = model.val(data=data_yaml, imgsz=cfg.imgsz, batch=cfg.batch, device=cfg.device, plots=False)
        print("[Eval] precision=", getattr(metrics.box, "mp", None))
        print("[Eval] recall=", getattr(metrics.box, "mr", None))
        print("[Eval] mAP50=", getattr(metrics.box, "map50", None))
        print("[Eval] mAP50-95=", getattr(metrics.box, "map", None))
    except Exception as exc:
        print(f"[WARN] validation summary failed: {exc}")


if __name__ == "__main__":
    try:
        import torch.multiprocessing as mp

        mp.freeze_support()
    except Exception:
        pass
    main()
