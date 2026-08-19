"""
Cluster-bootstrap 95% CIs for the idiom and per-category literal-rate deltas
(alpha=-4.78 vs. alpha=0.0), reported in the report's Table "Breakdown by
figurative category" and referenced in the Analysis section.

Resamples with replacement over expressions/idioms -- the true independent
sampling unit -- rather than over individual generations, since each
expression's 5 wording variants and each cell's 5 generated samples are
correlated, not independent draws. Treating all 125 generations per category
as i.i.d. would understate the true uncertainty (pseudo-replication).

Runs entirely off the committed labeled CSVs in results/; no GPU, no model
downloads.

Usage:
    python analysis/bootstrap_category_ci.py
"""

import numpy as np
import pandas as pd

RNG_SEED = 0
N_BOOTSTRAP = 5000
ALPHA_BASELINE = 0.0
ALPHA_LITERAL_PUSH = -4.78


def literal_rate(df: pd.DataFrame) -> float:
    return (df["label"] == "LITERAL").mean()


def bootstrap_delta(df: pd.DataFrame, group_col: str, rng: np.random.Generator,
                     n_bootstrap: int = N_BOOTSTRAP) -> np.ndarray:
    """95%-CI-ready bootstrap distribution for literal_rate(alpha=-4.78) -
    literal_rate(alpha=0.0), resampling whole groups (idioms/expressions)
    with replacement so correlated rows move together."""
    groups = df[group_col].unique()
    n = len(groups)
    baseline_by_group = {g: df[(df[group_col] == g) & (df.alpha_factor == ALPHA_BASELINE)] for g in groups}
    pushed_by_group = {g: df[(df[group_col] == g) & (df.alpha_factor == ALPHA_LITERAL_PUSH)] for g in groups}

    deltas = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(groups, size=n, replace=True)
        baseline = pd.concat([baseline_by_group[g] for g in sample])
        pushed = pd.concat([pushed_by_group[g] for g in sample])
        deltas[b] = literal_rate(pushed) - literal_rate(baseline)
    return deltas


def report_ci(name: str, df: pd.DataFrame, group_col: str, rng: np.random.Generator) -> dict:
    point = literal_rate(df[df.alpha_factor == ALPHA_LITERAL_PUSH]) - literal_rate(df[df.alpha_factor == ALPHA_BASELINE])
    deltas = bootstrap_delta(df, group_col, rng)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    n_groups = df[group_col].nunique()
    print(f"{name} (n={n_groups} {group_col}s): delta={point:+.3f}, 95% CI=[{lo:.3f}, {hi:.3f}]")
    return {"name": name, "n_groups": n_groups, "delta": point, "ci_lo": lo, "ci_hi": hi}


def main():
    idiom = pd.read_csv("results/labeled_idiom_results.csv")
    fig = pd.read_csv("results/labeled_figurative_results.csv")
    rng = np.random.default_rng(RNG_SEED)

    rows = [report_ci("Idiom", idiom, "idiom", rng)]
    for category in fig["category"].unique():
        rows.append(report_ci(category, fig[fig.category == category], "expression", rng))

    out = pd.DataFrame(rows)
    out.to_csv("analysis/bootstrap_category_ci.csv", index=False)
    print("\nSaved analysis/bootstrap_category_ci.csv")


if __name__ == "__main__":
    main()
