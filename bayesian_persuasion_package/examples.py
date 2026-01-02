
"""
examples.py

Example environments from Kamenica & Gentzkow (2011), adapted to the
finite-state, finite-action setup required by the LP engine.

We keep the paper's notation in docstrings:
- Ω: states
- A: actions
- μ0: prior
- u(a, ω): receiver payoff
- v(a, ω): sender payoff
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PersuasionInstance:
    """A complete finite Bayesian persuasion instance (Ω, A, μ0, u, v)."""
    name: str
    Omega: List[str]
    A: List[str]
    mu0: np.ndarray          # shape (|Ω|,)
    u: np.ndarray            # shape (|A|, |Ω|)
    v: np.ndarray            # shape (|A|, |Ω|)
    meta: Dict[str, Any]


def prosecutor_example(mu_guilty: float = 0.3) -> PersuasionInstance:
    """
    The binary-state prosecutor/judge example from the introduction of the paper.

    Paper description (Section I, motivating example):
      Ω = {innocent, guilty}
      A = {acquit, convict}
      μ0(guilty) = 0.3
      Receiver payoff: 1 if correct action, 0 if incorrect.
      Sender payoff: 1 if convict, 0 if acquit (independent of ω).

    Returns a PersuasionInstance with u and v coded exactly as above.
    """
    Omega = ["innocent", "guilty"]
    A = ["acquit", "convict"]

    mu0 = np.array([1.0 - mu_guilty, mu_guilty], dtype=float)

    # u(a, ω)
    # acquit correct if innocent; convict correct if guilty.
    u = np.array([
        [1.0, 0.0],  # acquit: u(acquit, innocent)=1, u(acquit, guilty)=0
        [0.0, 1.0],  # convict: u(convict, innocent)=0, u(convict, guilty)=1
    ], dtype=float)

    # v(a, ω): prosecutor only cares about conviction
    v = np.array([
        [0.0, 0.0],  # acquit
        [1.0, 1.0],  # convict
    ], dtype=float)

    meta = {"mu_guilty": mu_guilty}
    return PersuasionInstance(
        name="prosecutor_binary",
        Omega=Omega,
        A=A,
        mu0=mu0,
        u=u,
        v=v,
        meta=meta,
    )


def lobbying_example_discrete(
    n_states: int = 21,
    n_actions: int = 21,
    alpha: float = 0.75,
    omega_star: float = 1.2,
    prior: str = "uniform",
    beta_params: Tuple[float, float] = (2.0, 2.0),
) -> PersuasionInstance:
    """
    Discretized version of the lobbying example in Section V.A of the paper.

    Continuous model in the paper:
      ω ∈ [0,1] is socially optimal policy
      a ∈ [0,1] is politician's chosen policy
      u(a, ω) = -(a - ω)^2   (receiver)
      lobbyist prefers a* = α ω + (1-α) ω* with ω* > 1
      v(a, ω) = -(a - a*)^2  (sender)

    Paper result:
      - If α > 1/2: full disclosure is uniquely optimal
      - If α < 1/2: no disclosure is uniquely optimal
      - If α = 1/2: all signals equivalent

    Here we discretize ω on an n_states grid and a on an n_actions grid.
    This preserves the *finite-state, finite-action* structure needed for the LP.

    Args:
        n_states: |Ω| on an evenly spaced grid in [0,1]
        n_actions: |A| on an evenly spaced grid in [0,1]
        alpha: alignment parameter α
        omega_star: ω* > 1 (bias target)
        prior: "uniform" or "beta"
        beta_params: parameters (a,b) for Beta(a,b) if prior="beta"

    Returns:
        PersuasionInstance.
    """
    if n_states < 2 or n_actions < 2:
        raise ValueError("Need at least 2 states and 2 actions.")

    omega_grid = np.linspace(0.0, 1.0, n_states)
    action_grid = np.linspace(0.0, 1.0, n_actions)

    Omega = [f"{w:.3f}" for w in omega_grid]
    A = [f"{a:.3f}" for a in action_grid]

    # Prior μ0 over ω grid
    if prior == "uniform":
        mu0 = np.ones(n_states, dtype=float) / n_states
    elif prior == "beta":
        from scipy.stats import beta as beta_dist
        a_b, b_b = beta_params
        # Discretize by evaluating density at midpoints then normalize.
        # (This is simple and fine for illustration.)
        dens = beta_dist.pdf(omega_grid, a_b, b_b)
        dens = np.where(np.isfinite(dens), dens, 0.0)
        mu0 = dens / dens.sum()
    else:
        raise ValueError("prior must be 'uniform' or 'beta'.")

    # Build u and v arrays of shape (n_actions, n_states)
    u = np.zeros((n_actions, n_states), dtype=float)
    v = np.zeros((n_actions, n_states), dtype=float)

    for i_a, a in enumerate(action_grid):
        for i_w, w in enumerate(omega_grid):
            u[i_a, i_w] = - (a - w) ** 2
            a_star = alpha * w + (1.0 - alpha) * omega_star
            v[i_a, i_w] = - (a - a_star) ** 2

    meta = {
        "alpha": alpha,
        "omega_star": omega_star,
        "omega_grid": omega_grid,
        "action_grid": action_grid,
        "prior": prior,
        "beta_params": beta_params,
    }

    return PersuasionInstance(
        name=f"lobbying_discrete_alpha_{alpha:.2f}",
        Omega=Omega,
        A=A,
        mu0=mu0,
        u=u,
        v=v,
        meta=meta,
    )
