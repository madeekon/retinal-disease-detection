"""Report the Python / PyTorch / CUDA environment and save it as JSON.

Usage (from the project root):
    python scripts/00_check_environment.py
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retina.config import load_config, resolve_paths  # noqa: E402
from retina.utils import library_versions, save_json, setup_logging, timestamp  # noqa: E402


def torch_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    info = {
        "installed": True,
        "version": torch.__version__,
        "built_with_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update({
            "device_name": props.name,
            "total_vram_gb": round(props.total_memory / 1024 ** 3, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "bf16_supported": torch.cuda.is_bf16_supported(),
        })
        try:  # a tiny mixed-precision matmul proves that the GPU path works end to end
            a = torch.randn(256, 256, device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                b = a @ a
            torch.cuda.synchronize()
            info["fp16_autocast_test"] = f"ok ({b.dtype})"
        except Exception as exc:  # noqa: BLE001
            info["fp16_autocast_test"] = f"failed: {exc}"
    return info


def nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        res = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total,memory.used",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return res.stdout.strip() or res.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        return f"failed: {exc}"


def main() -> None:
    cfg = load_config()
    paths = resolve_paths(cfg)
    log = setup_logging(paths.logs_dir / "00_check_environment.log")
    ram_gb = None
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except ImportError:
        pass
    env = {
        "timestamp": timestamp(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "ram_gb": ram_gb,
        "libraries": library_versions(),
        "torch": torch_info(),
        "nvidia_smi": nvidia_smi(),
        "dataset_root_exists": paths.dataset_root.exists(),
        "image_dir_exists": paths.image_dir.exists(),
    }
    save_json(env, paths.logs_dir / "environment.json")
    for key in ("python_executable", "python_version", "platform", "ram_gb", "nvidia_smi",
                "dataset_root_exists", "image_dir_exists"):
        log.info("%-20s %s", key, env[key])
    log.info("%-20s %s", "torch", env["torch"])
    log.info("%-20s %s", "libraries", env["libraries"])
    if env["torch"].get("installed") and not env["torch"].get("cuda_available"):
        log.warning("PyTorch is installed but CUDA is not available: training will fall back to CPU. "
                    "Install the CUDA build (see README) to use the RTX 3050.")
    log.info("Saved %s", paths.logs_dir / "environment.json")


if __name__ == "__main__":
    main()
