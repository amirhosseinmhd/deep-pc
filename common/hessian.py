"""Hessian utilities for condition number computation."""

import jax.numpy as jnp


def unwrap_hessian_pytree(hessian_pytree, activities):
    """Convert Hessian pytree to dense matrix."""
    activities = activities[:-1]
    hessian_pytree = hessian_pytree[:-1]
    widths = [a.shape[1] for a in activities]
    N = sum(widths)
    hessian_matrix = jnp.zeros((N, N))

    start_row = 0
    for l, pytree_l in enumerate(hessian_pytree):
        start_col = 0
        for k, pytree_k in enumerate(pytree_l[:-1]):
            block = pytree_k[0, :, 0].reshape(widths[l], widths[k])
            hessian_matrix = hessian_matrix.at[
                start_row:start_row + widths[l],
                start_col:start_col + widths[k]
            ].set(block)
            start_col += widths[k]
        start_row += widths[l]
    return hessian_matrix
