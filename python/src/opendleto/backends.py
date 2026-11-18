"""Backend selection without implicit device transfers."""
from __future__ import annotations

import numpy as np


def tensor_backend(tensor, backend: str = "auto") -> str:
    if backend not in ("auto", "numpy", "torch"):
        raise ValueError("backend must be 'auto', 'numpy', or 'torch'")
    is_torch = type(tensor).__module__.startswith("torch")
    selected = ("torch" if is_torch else "numpy") if backend == "auto" else backend
    if selected == "torch":
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install opendleto[torch] for the PyTorch backend") from exc
        if not is_torch:
            raise TypeError("Torch backend requires a torch.Tensor; convert explicitly")
        if tensor.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
            raise TypeError("PyTorch tensor must have a floating or complex dtype")
    elif is_torch:
        raise TypeError("NumPy backend requires an ndarray; use tensor.detach().cpu().numpy() explicitly")
    elif not isinstance(tensor, np.ndarray):
        raise TypeError("Input must be a NumPy ndarray or PyTorch tensor")
    return selected
