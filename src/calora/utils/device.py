from __future__ import annotations

import torch

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_device(spec: str | torch.device) -> torch.device:
    if isinstance(spec, torch.device):
        return spec
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def resolve_dtype(spec: str | torch.dtype) -> torch.dtype:
    if isinstance(spec, torch.dtype):
        return spec
    try:
        return _DTYPES[spec.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown dtype {spec!r}; expected one of {sorted(_DTYPES)}") from exc


def set_matmul_tf32(enabled: bool) -> None:
    """Allow (or forbid) TF32 tensor cores for fp32 CUDA matmuls, process-wide.

    Only the *matmul* flag is touched. cuDNN convolutions already run in TF32
    under PyTorch's default (``torch.backends.cudnn.allow_tf32 = True``), and
    every run this project has reported ran with that default, so forcing it
    off here would change the numbers of paths that never asked for a change.
    Harmless without CUDA: the flag is simply never consulted.
    """
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)


def matmul_tf32_enabled() -> bool:
    return bool(torch.backends.cuda.matmul.allow_tf32)
