"""Hardware and runtime inspection utilities for Computer Use OOTB.

This module centralises hardware detection logic that used to live in various
scripts.  It exposes helpers that are reused by the Gradio UI as well as the
installation tooling so that we only maintain one set of heuristics for GPU/MPS
availability, dependency recommendations, and ShowUI model placement.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:  # Optional dependency, will be installed automatically by the new script.
    import psutil  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - best effort import
    psutil = None  # type: ignore

try:  # Optional dependency for NVIDIA telemetry.
    from pynvml import NVMLError, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex
    from pynvml import (  # type: ignore
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlDeviceGetUtilizationRates,
        nvmlInit,
        nvmlShutdown,
    )
except ModuleNotFoundError:  # pragma: no cover - best effort import
    NVMLError = None  # type: ignore
    nvmlDeviceGetCount = None  # type: ignore
    nvmlDeviceGetHandleByIndex = None  # type: ignore
    nvmlDeviceGetMemoryInfo = None  # type: ignore
    nvmlDeviceGetName = None  # type: ignore
    nvmlDeviceGetUtilizationRates = None  # type: ignore
    nvmlInit = None  # type: ignore
    nvmlShutdown = None  # type: ignore


@dataclass
class GPUInfo:
    """Information about a detected GPU device."""

    name: str
    total_memory_gb: float | None
    backend: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_memory_gb": self.total_memory_gb,
            "backend": self.backend,
        }


SHOWUI_BASE_PATH_CANDIDATES = [
    Path("./models/showui-2b"),
    Path("./showui-2b"),
]

SHOWUI_AWQ_PATH_CANDIDATES = [
    Path("./models/showui-2b-awq"),
    Path("./models/showui-2b-awq-4bit"),
    Path("./showui-2b-awq"),
    Path("./showui-2b-awq-4bit"),
]


def _run_command(command: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603, S607 - trusted command execution
        command,
        check=False,
        capture_output=True,
        text=True,
    )


def _query_nvidia_smi() -> List[GPUInfo]:
    """Collect GPU statistics via ``nvidia-smi`` if available."""

    if shutil.which("nvidia-smi") is None:
        return []

    result = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )

    if result.returncode != 0:
        return []

    devices = []
    for line in result.stdout.strip().splitlines():
        try:
            name, memory = [item.strip() for item in line.split(",")]
            devices.append(GPUInfo(name=name, total_memory_gb=float(memory) / 1024, backend="cuda"))
        except ValueError:
            continue
    return devices


def detect_accelerator() -> Dict[str, Any]:
    """Detect the best available accelerator on the host machine."""

    system = platform.system()
    architecture = platform.machine()

    torch_info: Dict[str, Any] = {"installed": False}
    devices: List[GPUInfo] = _query_nvidia_smi()
    backend: str = "cpu"
    mps_available = False

    try:
        import torch  # type: ignore

        torch_info = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
        }

        if torch.cuda.is_available():
            backend = "cuda"
            if not devices:
                for idx in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(idx)
                    memory = torch.cuda.get_device_properties(idx).total_memory / 1024**3
                    devices.append(GPUInfo(name=name, total_memory_gb=round(memory, 2), backend="cuda"))
        elif torch.backends.mps.is_available():
            backend = "mps"
            mps_available = True
            devices.append(GPUInfo(name="Apple Silicon", total_memory_gb=None, backend="mps"))

    except ModuleNotFoundError:
        torch_info = {"installed": False}
        if system == "Darwin" and architecture.lower() == "arm64":
            mps_available = True
            backend = "mps"
        elif devices:
            backend = "cuda"

    detected_memory_gb = max((d.total_memory_gb or 0 for d in devices), default=0)

    recommended_index_url = None
    if backend == "cuda":
        recommended_index_url = "https://download.pytorch.org/whl/cu121"
    elif backend == "cpu":
        recommended_index_url = "https://download.pytorch.org/whl/cpu"

    return {
        "system": system,
        "architecture": architecture,
        "backend": backend,
        "mps_available": mps_available,
        "torch": torch_info,
        "gpus": [device.to_dict() for device in devices],
        "recommended_torch_index_url": recommended_index_url,
        "detected_memory_gb": detected_memory_gb,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def recommend_showui_profile(info: Dict[str, Any]) -> Dict[str, Any]:
    """Return a recommended ShowUI preset based on detected hardware."""

    backend = info.get("backend", "cpu")
    memory_gb = info.get("detected_memory_gb") or 0

    if backend == "cuda":
        if memory_gb >= 16:
            preset = "Default (Maximum)"
            max_pixels = 1344
            awq = False
        elif memory_gb >= 10:
            preset = "Medium"
            max_pixels = 1024
            awq = False
        else:
            preset = "Minimal"
            max_pixels = 960
            awq = True
    elif backend == "mps":
        preset = "Medium"
        max_pixels = 1024
        awq = False
    else:
        preset = "Minimal"
        max_pixels = 896
        awq = True

    return {
        "preset": preset,
        "max_pixels": max_pixels,
        "awq_4bit": awq,
        "reason": {
            "backend": backend,
            "memory_gb": memory_gb,
        },
    }


def _safe_psutil_percent(func: Any) -> float | None:
    if psutil is None:
        return None
    try:
        return float(func())
    except Exception:  # pragma: no cover - best effort sampling
        return None


def gather_resource_metrics() -> Dict[str, Any]:
    """Collect lightweight CPU/GPU metrics for visualisation in the UI."""

    metrics: Dict[str, Any] = {
        "cpu_percent": _safe_psutil_percent(lambda: psutil.cpu_percent(interval=None)) if psutil else None,
        "memory_percent": _safe_psutil_percent(lambda: psutil.virtual_memory().percent) if psutil else None,
        "ram_gb": None,
        "ram_used_gb": None,
        "gpu_percent": None,
        "gpu_memory_percent": None,
        "gpu_memory_total_gb": None,
        "gpu_memory_used_gb": None,
    }

    if psutil is not None:
        try:
            mem = psutil.virtual_memory()
            metrics["ram_gb"] = round(mem.total / (1024**3), 2)
            metrics["ram_used_gb"] = round((mem.total - mem.available) / (1024**3), 2)
        except Exception:  # pragma: no cover
            pass

    if nvmlInit is not None and NVMLError is not None:
        try:  # pragma: no cover - requires NVIDIA GPU
            nvmlInit()
            device_count = nvmlDeviceGetCount()
            if device_count > 0:
                handle = nvmlDeviceGetHandleByIndex(0)
                util = nvmlDeviceGetUtilizationRates(handle)
                memory = nvmlDeviceGetMemoryInfo(handle)
                metrics["gpu_percent"] = float(util.gpu)
                metrics["gpu_memory_percent"] = (memory.used / memory.total) * 100 if memory.total else None
                metrics["gpu_memory_total_gb"] = round(memory.total / (1024**3), 2)
                metrics["gpu_memory_used_gb"] = round(memory.used / (1024**3), 2)
        except NVMLError:
            pass
        finally:
            try:
                nvmlShutdown()
            except Exception:
                pass

    return metrics


def build_performance_plot_data(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prepare bar-plot data from collected metrics."""

    data: List[Dict[str, Any]] = []
    if metrics.get("cpu_percent") is not None:
        data.append({"Resource": "CPU", "Utilisation": round(metrics["cpu_percent"], 2)})
    if metrics.get("memory_percent") is not None:
        data.append({"Resource": "RAM", "Utilisation": round(metrics["memory_percent"], 2)})
    if metrics.get("gpu_percent") is not None:
        data.append({"Resource": "GPU", "Utilisation": round(metrics["gpu_percent"], 2)})
    if metrics.get("gpu_memory_percent") is not None:
        data.append({"Resource": "VRAM", "Utilisation": round(metrics["gpu_memory_percent"], 2)})
    return data


def _directory_size_in_gb(path: Path) -> float:
    total_bytes = 0
    for file in path.rglob("*"):
        if file.is_file():
            total_bytes += file.stat().st_size
    return round(total_bytes / (1024**3), 2)


def check_showui_assets() -> Dict[str, Any]:
    """Inspect whether the ShowUI base and quantised models are available locally."""

    def _resolve(paths: Iterable[Path]) -> Optional[Path]:
        for candidate in paths:
            if candidate.exists():
                return candidate
        return None

    base_path = _resolve(SHOWUI_BASE_PATH_CANDIDATES)
    awq_path = _resolve(SHOWUI_AWQ_PATH_CANDIDATES)

    manifest = {
        "base": {
            "available": base_path is not None,
            "path": str(base_path) if base_path else None,
            "size_gb": _directory_size_in_gb(base_path) if base_path else None,
        },
        "awq": {
            "available": awq_path is not None,
            "path": str(awq_path) if awq_path else None,
            "size_gb": _directory_size_in_gb(awq_path) if awq_path else None,
        },
    }

    return manifest


def resolve_showui_model_path(awq: bool) -> str:
    """Return the preferred local path for the selected ShowUI weight format."""

    candidates = SHOWUI_AWQ_PATH_CANDIDATES if awq else SHOWUI_BASE_PATH_CANDIDATES
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve()) + "/"

    # Fall back to the primary candidate to keep backwards compatibility.
    default_path = candidates[0]
    default_path.mkdir(parents=True, exist_ok=True)
    return str(default_path.resolve()) + "/"


def summarise_recommendations(
    info: Dict[str, Any],
    recommendation: Dict[str, Any],
    model_manifest: Dict[str, Any],
) -> str:
    """Build a human readable summary of the recommended configuration."""

    lines = ["### ShowUI Environment Summary"]
    backend = info.get("backend", "cpu")
    memory = info.get("detected_memory_gb")
    lines.append(f"- **Detected backend:** {backend.upper()}")
    if memory:
        lines.append(f"- **Approx. VRAM:** {memory:.1f} GB")

    torch_info = info.get("torch", {})
    if torch_info.get("installed"):
        lines.append(f"- **PyTorch:** {torch_info.get('version')}")
    else:
        torch_cmd = "pip install torch torchvision torchaudio"
        index_url = info.get("recommended_torch_index_url")
        if index_url:
            torch_cmd += f" --index-url {index_url}"
        lines.append(f"- **PyTorch not found.** Install via `{torch_cmd}`.")

    preset = recommendation.get("preset")
    lines.append(f"- **Suggested preset:** `{preset}` → max pixels {recommendation.get('max_pixels')}.")
    if recommendation.get("awq_4bit"):
        lines.append("- Quantised AWQ weights are recommended for this device.")
        if not model_manifest["awq"]["available"]:
            lines.append("  - *Tip:* Download them via `python install_tools/install_showui.py --precision awq-4bit`." )
    else:
        if not model_manifest["base"]["available"]:
            lines.append("- Base FP16 weights missing. Download with `python install_tools/install_showui.py`." )

    return "\n".join(lines)


def dump_environment_report(path: Path, info: Dict[str, Any], recommendation: Dict[str, Any]) -> None:
    """Persist the detection results so other tools can reuse them."""

    payload = {
        "info": info,
        "recommendation": recommendation,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "python": sys.version,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


__all__ = [
    "detect_accelerator",
    "recommend_showui_profile",
    "gather_resource_metrics",
    "build_performance_plot_data",
    "check_showui_assets",
    "resolve_showui_model_path",
    "summarise_recommendations",
    "dump_environment_report",
]

