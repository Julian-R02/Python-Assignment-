
"""
plots.py

Plotting utilities for the examples and LP outputs.

Dependencies: numpy, matplotlib

We avoid seaborn and keep styling minimal and reproducible.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def concave_envelope_1d(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the *least concave majorant* (concave envelope) of points (x,y) with x increasing.

    Returns:
        hull_idx: indices of x,y that form the upper concave hull
        y_env: envelope values at each x (linear interpolation between hull points)

    Notes:
        This is a simple slope-monotonicity algorithm appropriate for 1D grids.
        It constructs a piecewise-linear concave function that lies weakly above all points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1D arrays of the same length.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")

    n = x.size
    hull = [0, 1] if n >= 2 else [0]

    def slope(i: int, j: int) -> float:
        return (y[j] - y[i]) / (x[j] - x[i])

    for k in range(2, n):
        hull.append(k)
        while len(hull) >= 3:
            i, j, k2 = hull[-3], hull[-2], hull[-1]
            if slope(i, j) < slope(j, k2) - 1e-15:  # slopes must be nonincreasing for concavity
                hull.pop(-2)
            else:
                break

    hull_idx = np.array(hull, dtype=int)

    # Build envelope by linear interpolation across hull segments
    y_env = np.empty_like(y)
    for seg in range(len(hull_idx) - 1):
        i = hull_idx[seg]
        j = hull_idx[seg + 1]
        xi, xj = x[i], x[j]
        yi, yj = y[i], y[j]
        mask = (x >= xi) & (x <= xj)
        # Linear interpolation
        t = (x[mask] - xi) / (xj - xi)
        y_env[mask] = yi + t * (yj - yi)

    return hull_idx, y_env


def plot_prosecutor_concavification(
    mu0_guilty: float,
    filepath: str,
    posterior_points: Optional[Dict[str, Any]] = None,
    grid_n: int = 501,
) -> None:
    """
    Plot v̂(μ) and its concave closure V(μ) for the prosecutor example.

    We identify μ with Pr(guilty), consistent with the paper's Figure 2.

    posterior_points: optional dict with keys:
        - "mu_vals": list of posterior μ values (Pr(guilty))
        - "weights": list of probabilities on those posteriors
    """
    mu_grid = np.linspace(0.0, 1.0, grid_n)

    # In prosecutor example, receiver convicts iff μ >= 0.5; sender payoff is 1 if convict else 0.
    v_hat = (mu_grid >= 0.5).astype(float)

    # Concave envelope on a grid (least concave majorant):
    _, V = concave_envelope_1d(mu_grid, v_hat)

    plt.figure()
    plt.plot(mu_grid, v_hat, label=r"$\hat v(\mu)$ (sender payoff under receiver best response)")
    plt.plot(mu_grid, V, label=r"$V(\mu)$ (concave closure / concavification)")
    plt.axvline(mu0_guilty, linestyle="--", label=r"prior $\mu_0$")
    plt.ylim(-0.05, 1.05)
    plt.xlim(0.0, 1.0)
    plt.xlabel(r"$\mu = \Pr(\mathrm{guilty})$")
    plt.ylabel("sender payoff")

    if posterior_points is not None:
        mu_vals = np.asarray(posterior_points.get("mu_vals", []), dtype=float)
        weights = np.asarray(posterior_points.get("weights", []), dtype=float)
        for m, w in zip(mu_vals, weights):
            plt.scatter([m], [1.0 if m >= 0.5 else 0.0])
            plt.annotate(f"μ={m:.3f}, wt={w:.3f}", (m, 1.0 if m >= 0.5 else 0.0),
                         textcoords="offset points", xytext=(5, 5))

    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


def plot_baseline_vs_optimal(
    labels: List[str],
    baseline: List[float],
    optimal: List[float],
    filepath: str,
    ylabel: str = "sender expected payoff",
    title: str = "Baseline vs Optimal (LP)",
) -> None:
    """Bar chart comparing baseline (no info) and optimal (LP) payoffs."""
    x = np.arange(len(labels))

    plt.figure()
    width = 0.35
    plt.bar(x - width / 2, baseline, width, label="baseline (no info)")
    plt.bar(x + width / 2, optimal, width, label="optimal (LP)")
    plt.xticks(x, labels, rotation=15)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


def plot_lobbying_expected_action(
    omega_grid: np.ndarray,
    action_grid: np.ndarray,
    pi: np.ndarray,
    baseline_action_value: float,
    filepath: str,
    title: str,
) -> None:
    """
    Plot E[a | ω] implied by the optimal signal (π), against ω.

    This is an intuitive "informativeness" visualization:
    - full disclosure would look like E[a|ω] ≈ ω
    - no disclosure would look like E[a|ω] ≈ constant (baseline action)
    """
    omega_grid = np.asarray(omega_grid, dtype=float)
    action_grid = np.asarray(action_grid, dtype=float)
    pi = np.asarray(pi, dtype=float)

    # Expected recommended action conditional on ω:
    # pi shape: (n_actions, n_states)
    E_a_given_omega = action_grid @ pi  # shape (n_states,)

    plt.figure()
    plt.plot(omega_grid, E_a_given_omega, label=r"$\mathbb{E}[a\mid \omega]$ under optimal $\pi$")
    plt.plot(omega_grid, omega_grid, linestyle="--", label="full disclosure benchmark (a=ω)")
    plt.axhline(baseline_action_value, linestyle=":", label="baseline (no info) action")
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"recommended action")
    plt.title(title)
    plt.ylim(min(action_grid) - 0.05, max(action_grid) + 0.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()
