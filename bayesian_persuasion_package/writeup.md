
# Bayesian Persuasion (Kamenica & Gentzkow, 2011) — finite Ω, finite A in Python

This folder contains a minimal, reproducible implementation of the **sender’s optimal information design** problem in the **finite-state, finite-action** Bayesian persuasion model.

The core computational object is a **linear program (LP)** that solves for the sender-optimal **straightforward signal** (a “recommended action” signal), consistent with **Proposition 1** of Kamenica & Gentzkow (2011): it is without loss to focus on distributions of posteriors that satisfy **Bayes plausibility**, and without loss to restrict to **straightforward** signals that recommend actions and are followed in equilibrium.

---

## 1. Model (paper notation)

- **States:** \( \Omega \) (finite), indexed by \( \omega \)
- **Actions:** \( A \) (finite), indexed by \( a \)
- **Prior:** \( \mu_0 \in \Delta(\Omega) \)
- **Receiver payoff:** \( u(a,\omega) \)
- **Sender payoff:** \( v(a,\omega) \)

Timing (as in the paper):

1. Sender commits to a signal \( \pi(\cdot \mid \omega) \)
2. Signal realization is observed by Receiver
3. Receiver chooses action \( a \)

Receiver is Bayesian and (weakly) best-responds to the posterior; ties are broken in the sender’s favor.

---

## 2. LP formulation (finite Ω, finite A)

### Decision variables

We use the standard “revelation principle” style formulation:

\[
x(a,\omega) \equiv \Pr(\text{recommend } a \ \text{and state } \omega).
\]

This is the *joint* distribution induced by the prior and the signal:
\[
x(a,\omega) = \mu_0(\omega)\,\pi(a\mid \omega).
\]

### Objective

Maximize sender expected payoff:

\[
\max_{x \ge 0} \ \sum_{a\in A}\sum_{\omega\in\Omega} v(a,\omega)\,x(a,\omega).
\]

### Constraints

**(i) Bayes plausibility / correct marginals**

Because the sender cannot change the prior distribution of states:

\[
\sum_{a\in A} x(a,\omega) = \mu_0(\omega) \quad \forall \omega\in\Omega.
\]

This is exactly the finite version of the paper’s Bayes-plausibility restriction
\( \mathbb{E}_\tau[\mu] = \mu_0 \) once we map \(x\) into posteriors.

**(ii) Receiver obedience (recommended action is optimal)**

Let \(p(a)=\sum_{\omega}x(a,\omega)\) be the probability that recommendation \(a\) is sent,
and define the posterior after seeing recommendation \(a\):

\[
\mu_a(\omega)=\Pr(\omega \mid a)=\frac{x(a,\omega)}{p(a)}.
\]

Obedience requires, for every recommended \(a\) and deviation \(a'\):

\[
\sum_{\omega} u(a,\omega)\,\mu_a(\omega) \ge \sum_{\omega} u(a',\omega)\,\mu_a(\omega).
\]

Multiplying by \(p(a)\) (which is \(\ge 0\)) yields a **linear** constraint in \(x\):

\[
\sum_{\omega} \big(u(a,\omega)-u(a',\omega)\big)\,x(a,\omega)\ge 0
\quad \forall a,a'\in A.
\]

### Recovering the signal

From \(x\), recover the conditional probabilities:

\[
\pi(a\mid \omega)=\frac{x(a,\omega)}{\mu_0(\omega)}.
\]

---

## 3. What’s in each file

- `solver.py`: the LP engine and utilities (baseline “no info” benchmark)
- `examples.py`: finite environments
  - prosecutor binary example (exactly as in the paper intro)
  - discretized lobbying example (paper Section V.A, discretized to finite Ω and A)
- `simulate.py`: Monte Carlo validation
- `plots.py`: figures (concavification and payoff comparisons)
- `main.py`: runs everything end-to-end and writes figures to `./outputs/`

---

## 4. Examples reproduced

### A) Prosecutor (binary state / binary action)

This reproduces the paper’s introductory example, including the optimal signal:

- \( \Pr(\text{guilty})=0.3 \)
- receiver chooses **convict** vs **acquit** with 0/1 correctness payoff
- sender wants conviction regardless of state

The LP recovers the classic optimal signal (paper eq. (1)) and achieves conviction with probability \(0.6\).

The script also produces a concavification-style plot of \( \hat v(\mu)\) and its concave closure \(V(\mu)\),
illustrating the posterior mixture that implements the optimum (LP remains the source of truth).

### B) Lobbying (discretized)

This is a finite-grid discretization of Section V.A:

- receiver wants \(a \approx \omega\)
- sender wants \(a \approx \alpha\omega + (1-\alpha)\omega^*\) with \(\omega^*>1\)

The paper’s sharp characterization is:
- if \( \alpha>1/2 \): **full disclosure**
- if \( \alpha<1/2 \): **no disclosure**

On a discrete grid, the LP reproduces this logic approximately:
- for \(\alpha=0.25\): optimal policy recommendations are nearly constant
- for \(\alpha=0.75\): recommendations track \(\omega\) closely (close to full disclosure)

---

## 5. Validation checks

`simulate.py` provides:

- **Exact checks** from the LP solution:
  - Bayes plausibility: \( \max_\omega |\sum_a x(a,\omega) - \mu_0(\omega)| \)
  - obedience slack: \( \min_{a,a'} \sum_\omega (u(a,\omega)-u(a',\omega))x(a,\omega) \)
  - sender value: \( \sum_{a,\omega} v(a,\omega)x(a,\omega) \)

- **Monte Carlo checks**:
  - sample \( \omega \sim \mu_0 \), then \( a \sim \pi(\cdot\mid\omega)\)
  - verify Bayes plausibility and obedience empirically
  - verify simulated sender payoff matches the LP objective up to sampling error

All randomness uses fixed seeds.

---

## 6. How to run

From this folder:

```bash
python main.py
```

Outputs are written to:

- `outputs/fig_prosecutor_concavification.png`
- `outputs/fig_lobbying_expected_action_alpha_0.25.png`
- `outputs/fig_lobbying_expected_action_alpha_0.75.png`
- `outputs/fig_payoffs_baseline_vs_optimal.png`
- `outputs/summary.json`
