"""Stub for scripts.setup_gpu — only used when --setup-gpu != none (default is none)."""


def install_target(target: str) -> int:
    print(f"[setup-gpu stub] install_target({target!r}) called — no-op.")
    return 0


def print_status() -> bool:
    try:
        import torch
        available = torch.cuda.is_available()
    except ImportError:
        available = False
    print(f"[setup-gpu stub] CUDA available: {available}")
    return available


def choose_target(mode: str | None) -> str:
    return mode or "cpu"
