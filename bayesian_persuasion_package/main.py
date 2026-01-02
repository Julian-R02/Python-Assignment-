
"""
main.py

End-to-end reproduction script:
- solves the core finite Bayesian persuasion LP
- runs 2 classic examples (prosecutor + discretized lobby)
- simulates to validate Bayes plausibility, obedience, and sender payoff
- generates plots

Run:
    python main.py

Outputs:
    ./outputs/*.png
    console logs with key numbers
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from solver import solve_persuasion_lp
from examples import prosecutor_example, lobbying_example_discrete
from simulate import simulate_from_signal, exact_bayes_plausibility_error, exact_obedience_min_slack, exact_sender_value
from plots import plot_prosecutor_concavification, plot_baseline_vs_optimal, plot_lobbying_expected_action


def fmt(x: float, nd: int = 6) -> str:
    return f"{x:.{nd}f}"


def run_prosecutor(out_dir: Path) -> dict:
    inst = prosecutor_example(mu_guilty=0.3)
    sol = solve_persuasion_lp(inst.mu0, inst.u, inst.v, A=inst.A, Omega=inst.Omega)

    assert sol.success, sol.message

    print("\n=== Prosecutor (binary) example ===")
    print("Prior μ0(guilty) =", fmt(inst.mu0[1]))
    print("Baseline (no info): receiver action =", inst.A[sol.a0_index])
    print("Baseline sender payoff =", fmt(sol.baseline_sender_value))
    print("Optimal sender payoff (LP) =", fmt(sol.objective_value))
    print("Unconditional Pr(recommend convict) =", fmt(sol.p_a[inst.A.index('convict')]))

    # Print π(a|ω) as in the paper's equation (1)
    # Mapping: message "i" ~ recommend acquit; message "g" ~ recommend convict.
    idx_acquit = inst.A.index("acquit")
    idx_convict = inst.A.index("convict")
    idx_innocent = inst.Omega.index("innocent")
    idx_guilty = inst.Omega.index("guilty")

    print("\nSignal π(a|ω):")
    print("  π(acquit | innocent) =", fmt(sol.pi[idx_acquit, idx_innocent]), " (paper: 4/7 ≈ 0.571429)")
    print("  π(acquit | guilty)   =", fmt(sol.pi[idx_acquit, idx_guilty]),   " (paper: 0)")
    print("  π(convict| innocent) =", fmt(sol.pi[idx_convict, idx_innocent]), " (paper: 3/7 ≈ 0.428571)")
    print("  π(convict| guilty)   =", fmt(sol.pi[idx_convict, idx_guilty]),   " (paper: 1)")

    print("\nPosteriors μ_a(guilty):")
    mu_acquit = sol.mu_a[idx_acquit, idx_guilty]
    mu_convict = sol.mu_a[idx_convict, idx_guilty]
    print("  μ_acquit(guilty)  =", fmt(mu_acquit),  " (paper: 0)")
    print("  μ_convict(guilty) =", fmt(mu_convict), " (paper: 0.5)")

    # Exact checks
    bayes_l1, bayes_max = exact_bayes_plausibility_error(sol.x, inst.mu0)
    obed_min = exact_obedience_min_slack(sol.x, inst.u)
    sender_val_exact = exact_sender_value(sol.x, inst.v)
    print("\nExact checks from LP solution:")
    print("  Bayes plausibility max abs error =", fmt(bayes_max))
    print("  Obedience min slack              =", fmt(obed_min))
    print("  Sender payoff (recomputed)       =", fmt(sender_val_exact))

    # Simulation
    sim = simulate_from_signal(inst.mu0, sol.pi, inst.u, inst.v, n=200_000, seed=123)
    print("\nSimulation (n=200k) checks:")
    print("  Bayes plausibility max abs error (empirical, vs μ0) =", fmt(sim.bayes_plausibility_max_abs_error))
    print("  Obedience min slack (empirical)              =", fmt(sim.obedience_min_slack))
    print("  Sender payoff (empirical)                    =", fmt(sim.empirical_sender_value))

    # Concavification plot
    mu_vals = [mu_acquit, mu_convict]
    weights = [sol.p_a[idx_acquit], sol.p_a[idx_convict]]
    fig_path = out_dir / "fig_prosecutor_concavification.png"
    plot_prosecutor_concavification(
        mu0_guilty=float(inst.mu0[idx_guilty]),
        filepath=str(fig_path),
        posterior_points={"mu_vals": mu_vals, "weights": weights},
    )
    print("\nSaved:", fig_path)

    return {
        "label": "prosecutor",
        "baseline_sender": sol.baseline_sender_value,
        "optimal_sender": sol.objective_value,
        "solution": sol,
        "instance": inst,
        "fig_concavification": str(fig_path),
    }


def run_lobby(out_dir: Path, alpha: float) -> dict:
    inst = lobbying_example_discrete(n_states=21, n_actions=21, alpha=alpha, omega_star=1.2, prior="uniform")
    sol = solve_persuasion_lp(inst.mu0, inst.u, inst.v, A=inst.A, Omega=inst.Omega)
    assert sol.success, sol.message

    omega_grid = inst.meta["omega_grid"]
    action_grid = inst.meta["action_grid"]
    baseline_action_value = float(action_grid[sol.a0_index])

    print(f"\n=== Lobbying example (discrete grid), alpha={alpha:.2f} ===")
    print("Baseline (no info) receiver action â(μ0) ≈", fmt(baseline_action_value))
    print("Baseline sender payoff =", fmt(sol.baseline_sender_value))
    print("Optimal sender payoff (LP) =", fmt(sol.objective_value))
    print("Gain from persuasion =", fmt(sol.objective_value - sol.baseline_sender_value))

    # How concentrated is the signal? (informal): number of actions recommended with positive prob.
    n_used = int(np.sum(sol.p_a > 1e-10))
    print("Number of recommended actions used (p(a)>1e-10):", n_used, "out of", len(inst.A))

    # Plot E[a|ω]
    fig_path = out_dir / f"fig_lobbying_expected_action_alpha_{alpha:.2f}.png"
    plot_lobbying_expected_action(
        omega_grid=omega_grid,
        action_grid=action_grid,
        pi=sol.pi,
        baseline_action_value=baseline_action_value,
        filepath=str(fig_path),
        title=f"Lobbying example (discrete): expected recommendation vs ω (alpha={alpha:.2f})",
    )
    print("Saved:", fig_path)

    # Simulation check
    sim = simulate_from_signal(inst.mu0, sol.pi, inst.u, inst.v, n=200_000, seed=456 + int(alpha*100))
    print("Simulation sender payoff (empirical) =", fmt(sim.empirical_sender_value))
    print("Simulation Bayes max abs error (vs μ0)=", fmt(sim.bayes_plausibility_max_abs_error))
    print("Simulation obedience min slack       =", fmt(sim.obedience_min_slack))

    return {
        "label": f"lobby(alpha={alpha:.2f})",
        "baseline_sender": sol.baseline_sender_value,
        "optimal_sender": sol.objective_value,
        "solution": sol,
        "instance": inst,
        "fig_expected_action": str(fig_path),
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    results = []
    results.append(run_prosecutor(out_dir))
    results.append(run_lobby(out_dir, alpha=0.25))
    results.append(run_lobby(out_dir, alpha=0.75))

    # Payoff comparison plot
    labels = [r["label"] for r in results]
    baseline = [r["baseline_sender"] for r in results]
    optimal = [r["optimal_sender"] for r in results]
    fig_payoffs = out_dir / "fig_payoffs_baseline_vs_optimal.png"
    plot_baseline_vs_optimal(labels, baseline, optimal, filepath=str(fig_payoffs),
                             title="Sender payoff: baseline (no info) vs optimal (LP)")
    print("\nSaved:", fig_payoffs)

    # Save a small JSON summary for convenience.
    summary = {
        r["label"]: {
            "baseline_sender": float(r["baseline_sender"]),
            "optimal_sender": float(r["optimal_sender"]),
            "gain": float(r["optimal_sender"] - r["baseline_sender"]),
        }
        for r in results
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved:", out_dir / "summary.json")


if __name__ == "__main__":
    main()
