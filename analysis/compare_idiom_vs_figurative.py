"""
Local, CPU-only comparison of idiom vs. figurative-benchmark steering results.

Runs entirely off the committed labeled CSVs in results/ -- no GPU, no model
downloads. Produces the tables and plot used in the report's "Results and
Analysis" section (Step 3 / Step 5 of the project plan).

Requires results/labeled_idiom_results.csv and results/labeled_figurative_results.csv
to exist (the second is produced by notebooks/step_3_eval_figurative_benchmark.ipynb).

Usage:
    python analysis/compare_idiom_vs_figurative.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = "results"
OUT_DIR = "analysis"

IDIOM_LABELED = os.path.join(RESULTS_DIR, "labeled_idiom_results.csv")
FIGURATIVE_LABELED = os.path.join(RESULTS_DIR, "labeled_figurative_results.csv")


def summarize(df: pd.DataFrame, group_cols):
    summary = df.groupby(group_cols)["label"].value_counts(normalize=True).unstack().fillna(0)
    for col in ["FIGURATIVE", "LITERAL", "INCOHERENT"]:
        if col not in summary.columns:
            summary[col] = 0
    summary["coherence_rate"] = summary["FIGURATIVE"] + summary["LITERAL"]
    return summary[["FIGURATIVE", "LITERAL", "INCOHERENT", "coherence_rate"]].round(3)


def main():
    if not os.path.isfile(FIGURATIVE_LABELED):
        raise SystemExit(
            f"{FIGURATIVE_LABELED} not found yet. Run "
            "notebooks/step_3_eval_figurative_benchmark.ipynb on a GPU runtime first, "
            "then pull the results back into this repo."
        )

    idiom_df = pd.read_csv(IDIOM_LABELED)
    fig_df = pd.read_csv(FIGURATIVE_LABELED)

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Table 1: idiom vs. figurative (pooled), literal rate by alpha -----
    idiom_by_alpha = summarize(idiom_df, ["alpha_factor"])
    fig_by_alpha = summarize(fig_df, ["alpha_factor"])

    combined = pd.DataFrame({
        "idiom_literal_rate": idiom_by_alpha["LITERAL"],
        "figurative_literal_rate": fig_by_alpha["LITERAL"],
        "idiom_coherence_rate": idiom_by_alpha["coherence_rate"],
        "figurative_coherence_rate": fig_by_alpha["coherence_rate"],
    })
    combined.to_csv(os.path.join(OUT_DIR, "summary_by_alpha.csv"))
    print("=== Literal rate by alpha_factor: idioms vs. figurative benchmark (pooled) ===")
    print(combined)
    print()

    # --- Table 2: figurative benchmark broken down by category -------------
    fig_by_category_alpha = summarize(fig_df, ["category", "alpha_factor"])
    fig_by_category_alpha.to_csv(os.path.join(OUT_DIR, "summary_by_category_alpha.csv"))
    print("=== Figurative benchmark: literal/figurative rate by category x alpha_factor ===")
    print(fig_by_category_alpha)
    print()

    # --- Table 3: gold subset only (one rigorously-validated variant per ----
    # expression) -- a higher-confidence read, less sensitive to any one
    # borderline prefix in the full 5-variants-per-expression set.
    if "gold" in fig_df.columns:
        gold_df = fig_df[fig_df["gold"]]
        gold_by_alpha = summarize(gold_df, ["alpha_factor"])
        gold_by_alpha.to_csv(os.path.join(OUT_DIR, "summary_gold_by_alpha.csv"))
        print(f"=== Figurative benchmark, gold subset only ({gold_df['expression'].nunique()} expressions) ===")
        print(gold_by_alpha)
        print()

    # --- Delta: literal rate at max literal-push alpha vs. unsteered -------
    min_alpha = min(idiom_df["alpha_factor"].unique())
    idiom_delta = idiom_by_alpha.loc[min_alpha, "LITERAL"] - idiom_by_alpha.loc[0.0, "LITERAL"]
    print(f"Idiom literal-rate delta (alpha={min_alpha} vs. 0.0): {idiom_delta:+.3f}")

    for category in fig_df["category"].unique():
        cat_summary = summarize(fig_df[fig_df["category"] == category], ["alpha_factor"])
        cat_delta = cat_summary.loc[min_alpha, "LITERAL"] - cat_summary.loc[0.0, "LITERAL"]
        print(f"{category} literal-rate delta (alpha={min_alpha} vs. 0.0): {cat_delta:+.3f}")

    if "gold" in fig_df.columns:
        gold_delta = gold_by_alpha.loc[min_alpha, "LITERAL"] - gold_by_alpha.loc[0.0, "LITERAL"]
        print(f"Gold-subset literal-rate delta (alpha={min_alpha} vs. 0.0): {gold_delta:+.3f}")
    print()

    # --- Plot: literal rate vs. alpha, idiom vs. each figurative category --
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(idiom_by_alpha.index, idiom_by_alpha["LITERAL"], marker="o", label="Idioms", linewidth=2)
    for category in sorted(fig_df["category"].unique()):
        cat_summary = summarize(fig_df[fig_df["category"] == category], ["alpha_factor"])
        ax.plot(cat_summary.index, cat_summary["LITERAL"], marker="o", label=category)
    ax.set_xlabel("alpha_factor (negative = steer toward literal)")
    ax.set_ylabel("literal rate")
    ax.set_title("Literal rate vs. steering coefficient: idioms vs. figurative categories")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, "literal_rate_vs_alpha.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
