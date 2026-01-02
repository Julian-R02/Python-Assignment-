
"""
simulate.py

Simulation / verification utilities for the Bayesian persuasion LP solution.

We verify:
1) Bayes plausibility: posteriors average back to μ0
2) Receiver obedience: recommended action is (weakly) optimal given the posterior
3) Sender payoff from simulation matches LP objective (within sampling error)

All checks are available in both:
- exact form (using LP solution x)
- Monte Carlo form (sampling ω ~ μ0 and a ~ π(·|ω))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SimulationResult:
    n: int
    seed: int

    empirical_mu0: np.ndarray          # Pr_hat(ω)
    empirical_p_a: np.ndarray          # Pr_hat(a)
    empirical_mu_a: np.ndarray         # μ_hat_a(ω) = Pr_hat(ω|a), shape (n_a, n_ω)
    empirical_sender_value: float

    bayes_plausibility_l1_error: float
    bayes_plausibility_max_abs_error: float
    obedience_min_slack: float         # min_a [ EU_u(follow) - best_alt EU_u(alt) ]
    details: Dict[str, Any]


def simulate_from_signal(
    mu0: np.ndarray,
    pi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    n: int = 200_000,
    seed: int = 0,
) -> SimulationResult:
    """
    Monte Carlo simulation: sample ω ~ μ0, then sample recommendation a ~ π(·|ω),
    receiver plays recommended a.

    Args:
        mu0: prior, shape (n_ω,)
        pi: conditional signal π(a|ω), shape (n_a, n_ω)
        u: receiver payoff, shape (n_a, n_ω)
        v: sender payoff, shape (n_a, n_ω)
        n: number of samples
        seed: RNG seed

    Returns:
        SimulationResult with empirical posteriors and checks.
    """
    mu0 = np.asarray(mu0, dtype=float)
    pi = np.asarray(pi, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    n_a, n_w = pi.shape
    rng = np.random.default_rng(seed)

    # Sample states ω
    w_samples = rng.choice(n_w, size=n, p=mu0)

    # For each ω draw recommendation a ~ π(·|ω)
    a_samples = np.empty(n, dtype=int)
    for w in range(n_w):
        idx = np.where(w_samples == w)[0]
        if idx.size == 0:
            continue
        a_samples[idx] = rng.choice(n_a, size=idx.size, p=pi[:, w])

    # Empirical counts
    counts_w = np.bincount(w_samples, minlength=n_w).astype(float)
    counts_a = np.bincount(a_samples, minlength=n_a).astype(float)
    empirical_mu0 = counts_w / n
    empirical_p_a = counts_a / n

    # Empirical joint counts counts_aw[a,w]
    counts_aw = np.zeros((n_a, n_w), dtype=float)
    for a in range(n_a):
        idx_a = np.where(a_samples == a)[0]
        if idx_a.size == 0:
            continue
        ws = w_samples[idx_a]
        counts_aw[a] = np.bincount(ws, minlength=n_w).astype(float)

    # Empirical posteriors μ_hat_a(ω) = Pr(ω|a)
    empirical_mu_a = np.full((n_a, n_w), np.nan, dtype=float)
    for a in range(n_a):
        if counts_a[a] > 0:
            empirical_mu_a[a] = counts_aw[a] / counts_a[a]

    # Sender payoff in simulation:
    empirical_sender_value = float(np.mean(v[a_samples, w_samples]))

    # Bayes plausibility check (against the *true* prior μ0):
    # Compute mix = sum_a p_hat(a) μ_hat_a and compare to μ0.
    mix = np.zeros(n_w, dtype=float)
    for a in range(n_a):
        if empirical_p_a[a] > 0:
            mix += empirical_p_a[a] * empirical_mu_a[a]
    bayes_err_vec = mix - mu0
    bayes_l1 = float(np.sum(np.abs(bayes_err_vec)))
    bayes_max = float(np.max(np.abs(bayes_err_vec)))

    # Obedience check using empirical posterior:
    # For each a with positive probability,
    #   EU_u(follow a) - max_{a' != a} EU_u(a') >= 0.
    obedience_slacks = []
    for a in range(n_a):
        if empirical_p_a[a] <= 0:
            continue
        mu_hat = empirical_mu_a[a]
        EU_actions = u @ mu_hat  # shape (n_a,)
        best_alt = np.max(np.delete(EU_actions, a)) if n_a > 1 else -np.inf
        slack = float(EU_actions[a] - best_alt)
        obedience_slacks.append(slack)

    obedience_min = float(np.min(obedience_slacks)) if obedience_slacks else 0.0

    details = {
        "counts_aw": counts_aw,
        "bayes_mix": mix,
        "bayes_err_vec": bayes_err_vec,
        "obedience_slacks": obedience_slacks,
    }

    return SimulationResult(
        n=n,
        seed=seed,
        empirical_mu0=empirical_mu0,
        empirical_p_a=empirical_p_a,
        empirical_mu_a=empirical_mu_a,
        empirical_sender_value=empirical_sender_value,
        bayes_plausibility_l1_error=bayes_l1,
        bayes_plausibility_max_abs_error=bayes_max,
        obedience_min_slack=obedience_min,
        details=details,
    )


def exact_bayes_plausibility_error(x: np.ndarray, mu0: np.ndarray) -> Tuple[float, float]:
    """Exact Bayes plausibility error from the LP joint distribution x(a,ω)."""
    x = np.asarray(x, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    err = x.sum(axis=0) - mu0
    return float(np.sum(np.abs(err))), float(np.max(np.abs(err)))


def exact_obedience_min_slack(x: np.ndarray, u: np.ndarray) -> float:
    """
    Exact obedience slack:
        min_{a,a'≠a} ∑_ω (u(a,ω)-u(a',ω)) x(a,ω).

    This is >= 0 for an obedient recommendation signal.
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    n_a, _ = x.shape
    min_slack = float("inf")
    for a in range(n_a):
        for ap in range(n_a):
            if ap == a:
                continue
            slack = float((u[a] - u[ap]) @ x[a])
            min_slack = min(min_slack, slack)
    return 0.0 if min_slack == float("inf") else float(min_slack)


def exact_sender_value(x: np.ndarray, v: np.ndarray) -> float:
    """Exact sender value implied by x: ∑_{a,ω} v(a,ω) x(a,ω)."""
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    return float(np.sum(v * x))
