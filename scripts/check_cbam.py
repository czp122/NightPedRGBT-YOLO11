from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ultralytics import YOLO
from rgbt_config import AppConfig


def main():
    cfg = AppConfig.from_env()
    model_yaml = str((ROOT / cfg.model_yaml).resolve())
    m = YOLO(model_yaml).model
    # count CBAM modules (direct or inside C3k2CBAM)
    names = [module.__class__.__name__ for module in m.modules()]
    cbam_count = sum(n == 'CBAM' for n in names)
    c3k2cbam_count = sum(n == 'C3k2CBAM' for n in names)
    print('Model YAML:', model_yaml)
    print('Total params:', sum(p.numel() for p in m.parameters()))
    print('C3k2CBAM blocks:', c3k2cbam_count)
    print('CBAM modules:', cbam_count)
    # show a few occurrences
    print('Example modules:')
    shown = 0
    for n, mod in m.named_modules():
        if mod.__class__.__name__ in {'C3k2CBAM', 'CBAM'}:
            print(' -', n, ':', mod.__class__.__name__)
            shown += 1
            if shown >= 10:
                break


if __name__ == '__main__':
    main()
