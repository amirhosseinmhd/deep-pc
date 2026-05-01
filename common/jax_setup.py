"""Configure JAX backend: use CUDA/GPU when available, fall back to CPU."""
import os
import sys

if "JAX_PLATFORMS" not in os.environ:
    if sys.platform == "darwin":
        # Metal backend is experimental and missing many primitives (e.g.
        # default_memory_space); force CPU on Mac unless the caller explicitly
        # opts in via JAX_PLATFORMS=METAL.
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        os.environ["JAX_PLATFORMS"] = ""  # let JAX auto-detect (CUDA → CPU fallback)
