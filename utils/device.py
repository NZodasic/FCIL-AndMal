"""Device resolution and hardware acceleration utility.

Provides unified device resolution supporting Apple Silicon GPU (Metal Performance
Shaders / MPS), Nvidia GPU (CUDA), and CPU fallback.
"""

import os
import platform
import logging
from typing import Any, Dict, Optional, Union
import torch

logger = logging.getLogger(__name__)


def configure_mps_environment() -> None:
    """Set sensible environment variables for Apple Silicon MPS execution."""
    if platform.system() == "Darwin":
        # Enable fallback to CPU for operations not natively implemented in MPS
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        # Disable artificial low allocation watermark to allow full use of unified memory
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def is_mps_available() -> bool:
    """Check if Apple Silicon Metal Performance Shaders (MPS) is available."""
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def is_cuda_available() -> bool:
    """Check if Nvidia CUDA is available."""
    return torch.cuda.is_available()


def clear_device_cache() -> None:
    """Clear accelerator cache for Nvidia CUDA or Apple Silicon MPS."""
    if is_cuda_available():
        torch.cuda.empty_cache()
    if is_mps_available() and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def resolve_device(device_name: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve a target device string or object to an available torch.device.

    Supported inputs:
      - 'auto' / None / '': auto-select CUDA -> MPS -> CPU
      - 'mps': Apple Silicon GPU (with fallback if unavailable)
      - 'cuda' / 'cuda:0': Nvidia GPU (with fallback if unavailable)
      - 'cpu': Standard CPU execution

    Returns:
        torch.device: Available PyTorch device.
    """
    configure_mps_environment()

    if isinstance(device_name, torch.device):
        return device_name

    d_str = str(device_name).lower().strip() if device_name is not None else ""

    if not d_str or d_str in ("auto", "none", "default"):
        if is_cuda_available():
            return torch.device("cuda")
        elif is_mps_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    if d_str.startswith("cuda"):
        if is_cuda_available():
            try:
                return torch.device(d_str)
            except (RuntimeError, ValueError):
                return torch.device("cuda")
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    if d_str.startswith("mps"):
        if is_mps_available():
            return torch.device("mps")
        logger.warning("MPS (Apple Silicon GPU) requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    try:
        return torch.device(d_str)
    except (RuntimeError, ValueError):
        logger.warning(f"Unrecognized device '{device_name}'. Falling back to CPU.")
        return torch.device("cpu")


def get_device_info() -> Dict[str, Any]:
    """Retrieve detailed platform and hardware accelerator information."""
    info: Dict[str, Any] = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": is_cuda_available(),
        "mps_available": is_mps_available(),
    }

    if info["cuda_available"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()
    elif info["mps_available"]:
        info["accelerator"] = "Apple Silicon GPU (MPS / Metal)"

    optimal = resolve_device("auto")
    info["optimal_device"] = str(optimal)
    return info

