
"""
solver.py

Core LP engine for the finite-state, finite-action Bayesian persuasion problem
(Kamenica & Gentzkow, 2011).

We implement the sender's optimal information design problem as a linear program
over a *straightforward signal* (revelation principle / Proposition 1 in the paper).

Notation (matching the paper):
- Ω: finite state space, |Ω| = n_ω
- A: finite action space, |A| = n_a
- μ0 ∈ Δ(Ω): common prior over states
- u(a, ω): receiver payoff
- v(a, ω): sender payoff

A *straightforward signal* recommends an action a ∈ A. The LP chooses the joint
distribution x(a, ω) = Pr(recommend a AND state ω). Since Pr(ω) is fixed at μ0(ω),
the constraints enforce ∑_a x(a, ω) = μ0(ω) for all ω.

Receiver obedience constraints:
Given recommendation a, the receiver must (weakly) prefer a to any deviation a'.
In x variables this becomes linear:
  ∑_ω [u(a, ω) - u(a', ω)] x(a, ω) >= 0   for all a, a' ∈ A.

Objective:
Maximize sender expected payoff:
  max_x ∑_{a,ω} v(a, ω) x(a, ω).

From an optimal x we recover the signal (conditional distribution) π(a|ω) via:
  π(a|ω) = x(a, ω) / μ0(ω).

We also recover the posterior after each recommended action a:
  μ_a(ω) = x(a, ω) / p(a)   where p(a) = ∑_ω x(a, ω).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class PersuasionSolution:
    """Container for an LP solution and derived objects."""
    status: str
    success: bool
    message: str

    objective_value: float  # sender value under optimal signal
    x: np.ndarray           # shape (n_a, n_ω): joint distribution Pr(a, ω) = μ0(ω) π(a|ω)
    pi: np.ndarray          # shape (n_a, n_ω): π(a|ω)
    p_a: np.ndarray         # shape (n_a,): Pr(recommend a)
    mu_a: np.ndarray        # shape (n_a, n_ω): posterior μ_a(ω) when message/recommendation is a

    # Baseline (no-information) benchmark:
    a0_index: int           # receiver default action index at μ0 (sender-preferred among best responses)
    baseline_sender_value: float
    baseline_receiver_value: float

    # Diagnostics:
    obedience_slack_min: float
    bayes_plausibility_max_abs_error: float

    # Problem meta:
    A: List[str]
    Omega: List[str]
    mu0: np.ndarray
    u: np.ndarray
    v: np.ndarray


def _validate_inputs(mu0: np.ndarray, u: np.ndarray, v: np.ndarray) -> Tuple[int, int]:
    mu0 = np.asarray(mu0, dtype=float)
    if mu0.ndim != 1:
        raise ValueError("mu0 must be a 1D array of length |Ω|.")
    if not np.isclose(mu0.sum(), 1.0):
        raise ValueError("mu0 must sum to 1.")
    if np.any(mu0 < -1e-12):
        raise ValueError("mu0 must be nonnegative.")
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape (|A|, |Ω|).")
    if u.ndim != 2:
        raise ValueError("u and v must be 2D arrays of shape (|A|, |Ω|).")
    n_a, n_w = u.shape
    if mu0.shape[0] != n_w:
        raise ValueError("mu0 length must match |Ω| (second dimension of u/v).")
    return n_a, n_w


def receiver_default_action(mu0: np.ndarray, u: np.ndarray, v: np.ndarray) -> Tuple[int, float, float]:
    """
    Receiver's default action â(μ0) with sender-preferred tie-breaking.

    Returns:
      (a0_index, receiver_value, sender_value) under no information.
    """
    mu0 = np.asarray(mu0, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    EU = u @ mu0  # shape (n_a,)
    max_EU = EU.max()
    best_actions = np.flatnonzero(np.isclose(EU, max_EU))
    if best_actions.size == 1:
        a0 = int(best_actions[0])
    else:
        # Sender-preferred tie-break among receiver best responses:
        EV_sender = v @ mu0
        a0 = int(best_actions[np.argmax(EV_sender[best_actions])])

    receiver_val = float((u[a0] @ mu0))
    sender_val = float((v[a0] @ mu0))
    return a0, receiver_val, sender_val


def solve_persuasion_lp(
    mu0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    A: Optional[List[str]] = None,
    Omega: Optional[List[str]] = None,
    solver: str = "highs",
    tol: float = 1e-9,
) -> PersuasionSolution:
    """
    Solve the finite-state, finite-action Bayesian persuasion problem via LP.

    Args:
        mu0: prior over Ω, shape (n_ω,)
        u: receiver payoff u(a, ω), shape (n_a, n_ω)
        v: sender payoff v(a, ω), shape (n_a, n_ω)
        A: optional action names (length n_a)
        Omega: optional state names (length n_ω)
        solver: scipy.linprog method ("highs" recommended)
        tol: tolerance used in diagnostics

    Returns:
        PersuasionSolution with optimal signal (π) and induced posteriors (μ_a).
    """
    n_a, n_w = _validate_inputs(mu0, u, v)
    mu0 = np.asarray(mu0, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if A is None:
        A = [f"a{j}" for j in range(n_a)]
    if Omega is None:
        Omega = [f"ω{i}" for i in range(n_w)]
    if len(A) != n_a:
        raise ValueError("Length of A must equal |A| = number of rows of u/v.")
    if len(Omega) != n_w:
        raise ValueError("Length of Omega must equal |Ω| = number of columns of u/v.")

    # Variables: x[a, ω] flattened in row-major order (a changes slowest? We'll pick (a, ω) contiguous in ω).
    # Index mapping: idx(a, ω) = a*n_w + ω
    n_vars = n_a * n_w

    # Objective: maximize sum v(a,ω) x(a,ω) -> minimize -v·x
    c = -v.reshape(-1)

    # Equality constraints: for each ω, sum_a x[a, ω] = mu0[ω]
    A_eq = np.zeros((n_w, n_vars))
    for w in range(n_w):
        for a in range(n_a):
            A_eq[w, a * n_w + w] = 1.0
    b_eq = mu0.copy()

    # Inequality (obedience): for each a, a' != a: sum_ω (u[a,ω]-u[a',ω]) x[a,ω] >= 0
    # Convert to A_ub x <= b_ub by multiplying by -1.
    rows = []
    rhs = []
    for a in range(n_a):
        for ap in range(n_a):
            if ap == a:
                continue
            row = np.zeros(n_vars)
            diff = u[a] - u[ap]  # length n_w
            row[a * n_w : (a + 1) * n_w] = -diff  # -sum diff * x <= 0
            rows.append(row)
            rhs.append(0.0)
    A_ub = np.vstack(rows) if rows else None
    b_ub = np.array(rhs) if rhs else None

    bounds = [(0.0, None)] * n_vars

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=solver)

    if not res.success:
        status = str(res.status)
        msg = res.message
        # Still attempt to construct empty solution for easier debugging.
        x = np.full((n_a, n_w), np.nan)
        pi = np.full((n_a, n_w), np.nan)
        p_a = np.full(n_a, np.nan)
        mu_a = np.full((n_a, n_w), np.nan)
        a0, r0, s0 = receiver_default_action(mu0, u, v)
        return PersuasionSolution(
            status=status,
            success=False,
            message=msg,
            objective_value=float("nan"),
            x=x,
            pi=pi,
            p_a=p_a,
            mu_a=mu_a,
            a0_index=a0,
            baseline_sender_value=s0,
            baseline_receiver_value=r0,
            obedience_slack_min=float("nan"),
            bayes_plausibility_max_abs_error=float("nan"),
            A=A,
            Omega=Omega,
            mu0=mu0,
            u=u,
            v=v,
        )

    x = res.x.reshape(n_a, n_w)
    objective_value = float(v.reshape(-1) @ res.x)

    # Recover π(a|ω) = x(a,ω)/μ0(ω)
    pi = np.zeros_like(x)
    for w in range(n_w):
        if mu0[w] > 0:
            pi[:, w] = x[:, w] / mu0[w]
        else:
            pi[:, w] = 0.0

    # Message probabilities p(a) and posteriors μ_a
    p_a = x.sum(axis=1)
    mu_a = np.zeros_like(x)
    for a in range(n_a):
        if p_a[a] > 0:
            mu_a[a, :] = x[a, :] / p_a[a]
        else:
            mu_a[a, :] = np.nan  # never sent

    # Baseline default action at prior
    a0, receiver_val0, sender_val0 = receiver_default_action(mu0, u, v)

    # Diagnostics:
    # Bayes plausibility: sum_a x[a, ω] should equal mu0[ω]
    bayes_err = float(np.max(np.abs(x.sum(axis=0) - mu0)))

    # Obedience slack: compute min over (a, a') of LHS = sum diff x[a,ω]
    min_slack = float("inf")
    for a in range(n_a):
        for ap in range(n_a):
            if ap == a:
                continue
            slack = float((u[a] - u[ap]) @ x[a])
            min_slack = min(min_slack, slack)
    if min_slack == float("inf"):
        min_slack = 0.0

    return PersuasionSolution(
        status=str(res.status),
        success=True,
        message=res.message,
        objective_value=objective_value,
        x=x,
        pi=pi,
        p_a=p_a,
        mu_a=mu_a,
        a0_index=a0,
        baseline_sender_value=sender_val0,
        baseline_receiver_value=receiver_val0,
        obedience_slack_min=min_slack,
        bayes_plausibility_max_abs_error=bayes_err,
        A=A,
        Omega=Omega,
        mu0=mu0,
        u=u,
        v=v,
    )
