import platform
from importlib import import_module

import torch


class UnsupportedBackendError(RuntimeError):
    """Raised when SSD is imported or initialized on an unsupported runtime."""


def optional_import(module_name: str):
    try:
        return import_module(module_name)
    except ModuleNotFoundError:
        return None


def runtime_summary() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_version": getattr(torch.version, "cuda", None),
    }


def format_runtime_summary() -> str:
    info = runtime_summary()
    return (
        f"platform={info['platform']}, python={info['python']}, torch={info['torch_version']}, "
        f"cuda_available={info['cuda_available']}, cuda_version={info['cuda_version']}, hip_version={info['hip_version']}"
    )


def require_cuda_backend(component: str, missing_modules: list[str] | None = None):
    missing_modules = missing_modules or []
    if missing_modules:
        joined = ", ".join(sorted(missing_modules))
        raise UnsupportedBackendError(
            f"{component} requires CUDA-oriented SSD dependencies that are not installed: {joined}. "
            f"Current runtime: {format_runtime_summary()}"
        )
    if not torch.cuda.is_available():
        raise UnsupportedBackendError(
            f"{component} requires a CUDA-capable PyTorch runtime. Current runtime: {format_runtime_summary()}"
        )


def compile_if_supported(fn):
    try:
        return torch.compile(fn)
    except Exception:
        return fn


def choose_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
