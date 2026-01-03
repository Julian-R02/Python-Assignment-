# Python Assignment — Replication of Baron (1982) + Self‑Financing Extension

This repository contains a Jupyter/Python replication of **Baron (1982), “Regulating a Monopolist with Unknown Costs”**, plus a small extension that studies what happens when transfers must satisfy an (expected) **self‑financing** constraint.

The project is written as two notebooks:
- `Replication Code.ipynb`: the full, runnable replication + figures + extension (this is the “code” deliverable).
- `Final.ipynb`: the written report / answers (no code).

The original paper PDF is included for reference: `Baron-RegulatingMonopolistUnknown-1982.pdf`.

## What’s in this repo

| File | What it is | When to open it |
|---|---|---|
| `Replication Code.ipynb` | Full implementation of the model + optimal policy construction (incl. ironing), replication checks, IC diagnostics, simulations, figures, and the self‑financing extension | To reproduce results / run code |
| `Final.ipynb` | Written summary of the paper, research question, assumptions, and discussion of results | To read the report |
| `Baron-RegulatingMonopolistUnknown-1982.pdf` | Source paper | For background / citation |

## What the code does (high level)

The replication notebook follows the structure of the paper and implements the main mechanism design objects:

1. **Model primitives**
   - Demand via an inverse demand curve `P(q)` / value function `V(q)` (linear and isoelastic examples).
   - Costs with a private cost type `theta` and fixed + variable components.
2. **Virtual/adjusted cost**
   - Computes the paper’s “virtual cost” object `z_a(theta)`, which depends on the type distribution `F(theta)` and density `f(theta)`.
3. **Ironing**
   - Enforces the required monotonicity by “ironing” the virtual cost in CDF space (convexification step), producing an ironed function `z_fa(theta)`.
4. **Optimal policy construction**
   - Builds the policy schedule `{p(theta), q(theta), r(theta), s(theta)}`: regulated price, implied quantity, shutdown rule, and transfers (using the envelope condition with minimal rents).
5. **Replication + diagnostics**
   - Runs a worked-example check from the paper, then checks **incentive compatibility** numerically (truth‑telling vs. profitable deviations).
6. **Simulation + figures**
   - Produces tables/plots across parameters and distributions.
7. **Extension: self‑financing constraint**
   - Imposes an expected budget rule that **scales down positive subsidies** so expected subsidy outflows are covered by expected tax revenues, then evaluates how incentives, best‑response reporting, welfare, and shutdown rates change.

## How to run / reproduce

### Requirements
- Python **3.x** (the notebooks were written with Python 3; on macOS use `python3`, not `python`)
- Jupyter (Lab or Notebook)
- Packages: `numpy`, `pandas`, `matplotlib` (and `scipy` recommended; the notebook includes a fallback if SciPy isn’t available)

### Option A: create a fresh virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas matplotlib scipy jupyter
jupyter lab
```
Open `Replication Code.ipynb` and run **Kernel → Restart & Run All**.

### Option B: use your existing Jupyter setup
Open `Replication Code.ipynb` and run the cells top‑to‑bottom. The first code cell contains a convenience `pip install ...` line if you need it.

## Notes on reproducibility
- The notebook is **self‑contained**: no external datasets are downloaded.
- Computations are **deterministic** (grid‑based), so reruns should match up to small floating‑point tolerance.
- If `scipy` is missing, the quantity step uses a simple grid‑search fallback (slower / less accurate).

## Glossary (variables used throughout)
- `theta`: the firm’s private cost type (higher `theta` = higher cost)
- `F(theta)`, `f(theta)`: distribution and density of `theta`
- `a`: regulator weight on firm profit / rents (the policy varies with `a`)
- `p(theta)`: regulated price schedule
- `q(theta)`: implied quantity schedule
- `r(theta) in {0,1}`: operate vs. shutdown decision
- `s(theta)`: transfer (positive = subsidy, negative = tax)

## Reference
Baron (1982), *Regulating a Monopolist with Unknown Costs*.
