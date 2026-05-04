"""Auto-configure JAX platforms: prefer GPU (cuda) if available, fall back to CPU.

This module is imported early by entry scripts. It sets the environment
variable JAX_PLATFORMS appropriately so JAX will use CUDA when present.
"""
import os
import shutil


def _cuda_available():
    # Check common signals: CUDA_VISIBLE_DEVICES, nvidia-smi, /dev/nvidia0
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        return True
    if shutil.which("nvidia-smi") is not None:
        return True
    if os.path.exists("/dev/nvidia0"):
        return True
    return False


def configure_jax_platforms():
    # If the user already set JAX_PLATFORMS, respect it.
    if os.environ.get("JAX_PLATFORMS"):
        return

    if _cuda_available():
        # Allow both gpu and cpu as backends; JAX should pick gpu devices.
        os.environ["JAX_PLATFORMS"] = "cuda"
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"


# Configure on import
configure_jax_platforms()
