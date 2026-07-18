from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_env_file(path: Path) -> dict[str, str]:
    d: dict[str, str] = {}
    if not path.exists():
        return d
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        d[k.strip()] = v.strip().strip('"').strip("'")
    return d


def _get_bool(v: str, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _auto_device(v: str) -> str:
    raw = (v or "").strip().lower()
    if raw and raw != "auto":
        return v
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def _auto_workers(v: int) -> int:
    if v >= 0:
        return v
    cpu_count = os.cpu_count() or 4
    return max(1, min(8, cpu_count - 1))


@dataclass
class AppConfig:
    llvip_root: str = ""
    train_split: str = "visible/train/images"
    val_split: str = "visible/val/images"
    test_split: str = "visible/test/images"
    model_yaml: str = "configs/yolo11n_llvip_rgbt_cbam.yaml"
    weights: str = ""
    data_yaml_out: str = "runs/llvip_rgbt_data.yaml"
    imgsz: int = 640
    epochs: int = 20
    batch: int = 8
    workers: int = -1
    device: str = "auto"
    lr0: float = 1e-3
    weight_decay: float = 5e-4
    optimizer: str = "auto"
    amp: bool = True
    cache: bool = False
    patience: int = 30
    plots: bool = True
    audit_data: bool = True
    strict_data_audit: bool = False
    resume: bool = False
    seed: int = 0
    project: str = "runs"
    name: str = "llvip_yolo11n_rgbt_cbam"
    save_dir: str = "runs/rgbt_yolo11n_cbam"

    @classmethod
    def from_env(cls, env_path: str | None = None) -> "AppConfig":
        root = Path(__file__).resolve().parent
        p = Path(env_path) if env_path else (root / ".env")
        e = _parse_env_file(p)

        def gi(key: str, default: int) -> int:
            try:
                return int(e.get(key, str(default)))
            except Exception:
                return default

        def gf(key: str, default: float) -> float:
            try:
                return float(e.get(key, str(default)))
            except Exception:
                return default

        return cls(
            llvip_root=e.get("LLVIP_ROOT", ""),
            train_split=e.get("TRAIN_SPLIT", cls.train_split),
            val_split=e.get("VAL_SPLIT", cls.val_split),
            test_split=e.get("TEST_SPLIT", cls.test_split),
            model_yaml=e.get("MODEL_YAML", cls.model_yaml),
            weights=e.get("WEIGHTS", cls.weights),
            data_yaml_out=e.get("DATA_YAML_OUT", cls.data_yaml_out),
            imgsz=gi("IMGSZ", cls.imgsz),
            epochs=gi("EPOCHS", cls.epochs),
            batch=gi("BATCH", cls.batch),
            workers=_auto_workers(gi("WORKERS", cls.workers)),
            device=_auto_device(e.get("DEVICE", cls.device)),
            lr0=gf("LR0", cls.lr0),
            weight_decay=gf("WEIGHT_DECAY", cls.weight_decay),
            optimizer=e.get("OPTIMIZER", cls.optimizer),
            amp=_get_bool(e.get("AMP", str(cls.amp)), cls.amp),
            cache=_get_bool(e.get("CACHE", str(cls.cache)), cls.cache),
            patience=gi("PATIENCE", cls.patience),
            plots=_get_bool(e.get("PLOTS", str(cls.plots)), cls.plots),
            audit_data=_get_bool(e.get("AUDIT_DATA", str(cls.audit_data)), cls.audit_data),
            strict_data_audit=_get_bool(
                e.get("STRICT_DATA_AUDIT", str(cls.strict_data_audit)),
                cls.strict_data_audit,
            ),
            resume=_get_bool(e.get("RESUME", str(cls.resume)), cls.resume),
            seed=gi("SEED", cls.seed),
            project=e.get("PROJECT", cls.project),
            name=e.get("NAME", cls.name),
            save_dir=e.get("SAVE_DIR", cls.save_dir),
        )


def resolve_path(llvip_root: str, rel: str) -> str:
    if not llvip_root:
        return rel
    return str(Path(llvip_root) / rel)
