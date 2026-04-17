"""Shared utilities for rec-LRA variants (Ororbia & Mali, AAAI 2023).

Implements three appendix-only details that are required to replicate the
paper's CIFAR-10 / ImageNet numbers but were absent from the initial port:

- alpha_mix_error : convex combination of (z-y) and sign(z-y) with the
  α_e values reported on p.15 (0.19 for skip-error endpoints, 0.24 for
  adjacent-error endpoints).
- reproject_to_ball : Gaussian-ball gradient re-projection (p.16) used
  in *every* experiment in the paper to stabilise updates.
- add_input_noise : additive Gaussian σ=0.1 perturbation applied to the
  forward inputs during training (p.15).
"""

import jax.numpy as jnp
import jax.random as jr


def alpha_mix_error(z, y, alpha):
    """Return α (z − y) + (1 − α) sign(z − y).

    α=1.0 reduces to plain MSE-style errors (backwards-compat).
    α=0.19 / 0.24 are the paper's tuned values for the two endpoint kinds.
    """
    diff = z - y
    if alpha >= 1.0:
        return diff
    return alpha * diff + (1.0 - alpha) * jnp.sign(diff)


def reproject_to_ball(delta, c):
    """Project an update tensor back into a Gaussian ball of radius c.

    Nm(Δ, c) = (c / ||Δ||) Δ      if ||Δ|| ≥ c
              Δ                    otherwise
    Pass c=None or c<=0 to disable.
    """
    if c is None or c <= 0.0:
        return delta
    norm = jnp.linalg.norm(delta)
    scale = jnp.where(norm > c, c / (norm + 1e-12), 1.0)
    return delta * scale


def add_input_noise(key, x, sigma):
    """Add Gaussian noise of stdev σ to a batch tensor. σ=0 returns x."""
    if sigma is None or sigma <= 0.0:
        return x
    return x + sigma * jr.normal(key, x.shape, dtype=x.dtype)
