"""
Cluster-bootstrap 95% CIs for the idiom and per-category literal-rate deltas
(alpha=-4.78 vs. alpha=0.0), reported in the report's Table "Breakdown by
figurative category" and referenced in the Analysis section.

Resamples with replacement over expressions/idioms -- the true independent
sampling unit -- rather than over individual generations, since each
expression's 5 wording variants and each cell's 5 generated samples are
correlated, not independent draws. Treating all 125 generations per category
as i.i.d. would understate the true uncertainty (pseudo-replication).

Also reports paired-difference bootstrap CIs (e.g. delta_Idiom - delta_ConvMetaphor)
for the specific pairs the report's Analysis section compares. This is the correct
way to ask "are these two deltas distinguishable?" -- checking whether two marginal
CIs merely overlap is a known-conservative heuristic (it can miss real differences)
and is not equivalent to testing the difference directly. Each pair is resampled
independently (idioms and figurative-category expressions are disjoint item pools),
so delta_A_b - delta_B_b is a valid draw from the difference's sampling distribution
for every bootstrap replicate b.

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


def report_ci(name: str, deltas: np.ndarray, point: float, n_groups: int) -> dict:
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"{name} (n={n_groups}): delta={point:+.3f}, 95% CI=[{lo:.3f}, {hi:.3f}]")
    return {"name": name, "n_groups": n_groups, "delta": point, "ci_lo": lo, "ci_hi": hi}


def report_pairwise_diff(name_a: str, deltas_a: np.ndarray, name_b: str, deltas_b: np.ndarray) -> dict:
    """Paired-difference bootstrap: independently resampled deltas_a - deltas_b,
    per replicate. If the resulting 95% CI excludes 0, the two deltas are
    statistically distinguishable at that level -- the correct test, unlike
    eyeballing whether the two marginal CIs happen to overlap."""
    diff = deltas_a - deltas_b
    lo, hi = np.percentile(diff, [2.5, 97.5])
    distinguishable = not (lo <= 0 <= hi)
    print(f"{name_a} vs {name_b}: diff-in-delta={np.mean(diff):+.3f}, 95% CI=[{lo:.3f}, {hi:.3f}], "
          f"distinguishable={distinguishable}")
    return {"pair": f"{name_a} vs {name_b}", "diff": float(np.mean(diff)), "ci_lo": lo, "ci_hi": hi,
            "distinguishable": distinguishable}


def main():
    idiom = pd.read_csv("results/labeled_idiom_results.csv")
    fig = pd.read_csv("results/labeled_figurative_results.csv")
    rng = np.random.default_rng(RNG_SEED)

    all_deltas = {}
    rows = []

    d = bootstrap_delta(idiom, "idiom", rng)
    point = literal_rate(idiom[idiom.alpha_factor == ALPHA_LITERAL_PUSH]) - literal_rate(idiom[idiom.alpha_factor == ALPHA_BASELINE])
    all_deltas["Idiom"] = d
    rows.append(report_ci("Idiom", d, point, idiom["idiom"].nunique()))

    for category in fig["category"].unique():
        sub = fig[fig.category == category]
        d = bootstrap_delta(sub, "expression", rng)
        point = literal_rate(sub[sub.alpha_factor == ALPHA_LITERAL_PUSH]) - literal_rate(sub[sub.alpha_factor == ALPHA_BASELINE])
        all_deltas[category] = d
        rows.append(report_ci(category, d, point, sub["expression"].nunique()))

    out = pd.DataFrame(rows)
    out.to_csv("analysis/bootstrap_category_ci.csv", index=False)
    print("\nSaved analysis/bootstrap_category_ci.csv")

    print("\n=== Paired differences (the actual test for 'are these distinguishable?') ===")
    pairs = [
        ("Idiom", "Conventional Metaphor"),
        ("Conventional Metaphor", "Novel Metaphor"),
        ("Simile", "Conventional Metaphor"),
        ("Simile", "Idiom"),
    ]
    diff_rows = [report_pairwise_diff(a, all_deltas[a], b, all_deltas[b]) for a, b in pairs]
    pd.DataFrame(diff_rows).to_csv("analysis/bootstrap_pairwise_diff.csv", index=False)
    print("\nSaved analysis/bootstrap_pairwise_diff.csv")


if __name__ == "__main__":
    main()
