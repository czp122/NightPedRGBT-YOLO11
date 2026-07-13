from pathlib import Path


root = Path(SPECPATH)
datas = [
    (str(root / "yolo11n.pt"), "."),
    (str(root / "packaging" / "使用说明.txt"), "."),
    (
        str(root / "runs" / "rgbt_yolo11n_cbam11_ft8002" / "weights" / "best.pt"),
        "models",
    ),
]

a = Analysis(
    [str(root / "gui" / "app.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=["cv2_enumerate_cameras"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["onnx", "onnxruntime", "openvino", "tensorflow"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NightPedestrianDetection",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="NightPedestrianDetection",
)
