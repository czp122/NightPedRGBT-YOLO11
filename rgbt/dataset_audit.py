from __future__ import annotations

from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _paired_ir_path(visible_path: Path) -> Path:
    parts = ["infrared" if part.lower() == "visible" else part for part in visible_path.parts]
    return Path(*parts)


def audit_llvip_dataset(llvip_root: str | Path, splits: dict[str, str]) -> dict:
    """Inventory LLVIP pairs and detect sample leakage between configured splits."""
    root = Path(llvip_root).expanduser().resolve()
    report: dict = {"root": str(root), "splits": {}, "overlaps": {}}
    split_ids: dict[str, set[str]] = {}

    for name, configured_path in splits.items():
        image_dir = Path(configured_path).expanduser()
        if not image_dir.is_absolute():
            image_dir = root / image_dir
        image_dir = image_dir.resolve()
        images = sorted(
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ) if image_dir.is_dir() else []
        label_dir = image_dir.parent / "labels"
        identifiers: set[str] = set()
        missing_ir: list[str] = []
        missing_labels: list[str] = []

        for image_path in images:
            relative = image_path.relative_to(image_dir)
            identifier = relative.with_suffix("").as_posix().lower()
            identifiers.add(identifier)
            if not _paired_ir_path(image_path).is_file():
                missing_ir.append(relative.as_posix())
            if not (label_dir / relative).with_suffix(".txt").is_file():
                missing_labels.append(relative.as_posix())

        split_ids[name] = identifiers
        report["splits"][name] = {
            "path": str(image_dir),
            "images": len(images),
            "missing_ir": len(missing_ir),
            "missing_labels": len(missing_labels),
            "missing_ir_examples": missing_ir[:5],
            "missing_label_examples": missing_labels[:5],
        }

    split_names = list(splits)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            duplicate_ids = sorted(split_ids[left] & split_ids[right])
            report["overlaps"][f"{left}-{right}"] = {
                "count": len(duplicate_ids),
                "examples": duplicate_ids[:5],
            }
    return report


def print_dataset_audit(report: dict, *, strict: bool = False) -> None:
    """Print an actionable audit summary and optionally reject unsafe datasets."""
    problems: list[str] = []
    print(f"[DataAudit] root={report['root']}")
    for name, split in report["splits"].items():
        print(
            f"[DataAudit] {name}: images={split['images']}, "
            f"missing_ir={split['missing_ir']}, missing_labels={split['missing_labels']}"
        )
        if split["images"] == 0:
            problems.append(f"{name} contains no images")
        if split["missing_ir"]:
            problems.append(f"{name} has {split['missing_ir']} missing IR pairs")
        if split["missing_labels"]:
            problems.append(f"{name} has {split['missing_labels']} missing labels")

    for pair, overlap in report["overlaps"].items():
        if overlap["count"]:
            examples = ", ".join(overlap["examples"])
            message = f"{pair} share {overlap['count']} sample IDs"
            if examples:
                message += f" (for example: {examples})"
            problems.append(message)

    for problem in problems:
        print(f"[DataAudit][WARN] {problem}")
    if strict and problems:
        raise ValueError("Dataset audit failed in strict mode: " + "; ".join(problems))
