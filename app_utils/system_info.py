from __future__ import annotations

import os
import platform
import re
import sys
from typing import Any


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _format_windows_version(
    product_name: str,
    display_version: str,
    build: int,
    ubr: int | None = None,
    architecture: str = "",
) -> str:
    """Format Windows identity while correcting the legacy Windows 10 label."""
    product = _clean(product_name) or "Windows"
    if int(build or 0) >= 22000 and re.match(r"^Windows 10\b", product, re.IGNORECASE):
        product = re.sub(r"^Windows 10\b", "Windows 11", product, count=1, flags=re.IGNORECASE)

    parts = [product]
    version = _clean(display_version)
    if version and version.lower() not in product.lower():
        parts.append(version)

    build_text = str(int(build)) if int(build or 0) > 0 else "Unknown"
    if ubr is not None and int(ubr) >= 0 and build_text != "Unknown":
        build_text += f".{int(ubr)}"
    suffix = f"Build {build_text}"
    machine = _clean(architecture)
    if machine:
        suffix += f", {machine}"
    return f"{' '.join(parts)} ({suffix})"


def _windows_registry_values() -> dict[str, Any]:
    if os.name != "nt":
        return {}
    try:
        import winreg

        access = winreg.KEY_READ | getattr(winreg, "KEY_WOW64_64KEY", 0)
        key_path = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, access) as key:
            values = {}
            for name in ("ProductName", "DisplayVersion", "ReleaseId", "CurrentBuildNumber", "UBR"):
                try:
                    values[name] = winreg.QueryValueEx(key, name)[0]
                except OSError:
                    pass
            return values
    except (ImportError, OSError):
        return {}


def get_os_display_name() -> str:
    """Return the OS running the process, with reliable Windows 10/11 detection."""
    if os.name != "nt":
        return platform.platform()

    registry = _windows_registry_values()
    try:
        runtime_build = int(sys.getwindowsversion().build)
    except (AttributeError, TypeError, ValueError):
        runtime_build = 0
    try:
        registry_build = int(registry.get("CurrentBuildNumber", 0))
    except (TypeError, ValueError):
        registry_build = 0
    build = max(runtime_build, registry_build)
    try:
        ubr = int(registry["UBR"]) if "UBR" in registry else None
    except (TypeError, ValueError):
        ubr = None

    product_name = _clean(registry.get("ProductName")) or platform.win32_ver()[0] or "Windows"
    display_version = _clean(registry.get("DisplayVersion") or registry.get("ReleaseId"))
    return _format_windows_version(
        product_name,
        display_version,
        build,
        ubr,
        platform.machine(),
    )


def get_cpu_name() -> str:
    """Return a useful CPU model in frozen Windows apps where platform.processor is often blank."""
    if os.name == "nt":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                value = _clean(winreg.QueryValueEx(key, "ProcessorNameString")[0])
                if value:
                    return value
        except (ImportError, OSError):
            pass

    for value in (
        platform.processor(),
        os.environ.get("PROCESSOR_IDENTIFIER", ""),
        platform.machine(),
    ):
        cleaned = _clean(value)
        if cleaned:
            return cleaned
    return "Unknown"
