# Analysis: does the idiom steering vector generalize to non-idiomatic figurative language?

Generated from `results/labeled_idiom_results.csv` (Qwen/Qwen3-14B judge, 100-item
idiom benchmark) and `results/labeled_figurative_results.csv` (Qwen/Qwen3-14B judge,
100-item figurative benchmark) via `analysis/compare_idiom_vs_figurative.py`, after
both benchmarks were rebuilt to 100 items each (20 idioms / 20 expressions x 5
variants), matching IdioSteer's own benchmark size exactly, and passed the
dual-annotator validation protocol in `validation/annotator_prompt.md`.

This supersedes the analysis in `analysis/archive/findings.md`, which was run
against a smaller (41-row) idiom draft and a first-draft figurative benchmark that
have since been superseded.

## 1. Headline numbers (literal rate, pooled)

| alpha_factor | idiom literal rate | figurative literal rate |
|---:|---:|---:|
| -4.78 | 0.602 | 0.696 |
| -3.59 | 0.618 | 0.662 |
| -2.39 | 0.618 | 0.658 |
| -1.20 | 0.576 | 0.678 |
|  0.00 | 0.436 | 0.580 |
|  1.20 | 0.436 | 0.530 |
|  2.39 | 0.416 | 0.554 |
|  3.59 | 0.386 | 0.544 |
|  4.78 | 0.434 | 0.546 |

n = 500/alpha for both datasets (100 items x 5 samples). Coherence rate stays
>=0.90 in every cell for both datasets (full table: `analysis/summary_by_alpha.csv`),
so steering isn't degrading fluency.

**Both datasets now show a clean, bidirectional dose-response**: every negative-alpha
cell sits above its dataset's baseline (alpha=0) and every positive-alpha cell sits at
or below it, for both idioms and the figurative benchmark. This is a much cleaner
signal than the previous pass (41-row idiom draft), where the idiom curve was
non-monotonic and the baseline sat near 0.50. **Idiom delta (alpha=-4.78 vs 0.0):
+0.166. Figurative delta: +0.116.** Absolute baselines (0.436 idiom, 0.580 figurative)
are still far from the paper's reported ~0.12 idiom baseline, but the shape of the
effect (monotonic-ish, bidirectional, high-coherence) now qualitatively matches
the paper far better than before.

## 2. By figurative category (delta = alpha -4.78 vs 0.0, literal rate)

| category | delta | baseline literal rate | literal rate @ -4.78 |
|---|---:|---:|---:|
| Simile | +0.256 | 0.488 | 0.744 |
| Conventional Metaphor | +0.108 | 0.548 | 0.656 |
| Novel Metaphor | -0.008 (flat) | 0.736 | 0.728 |

- **Simile** shows the strongest transfer of any category: a bigger literal-rate
  swing than idioms themselves (+0.256 vs +0.166).
- **Conventional Metaphor** transfers in the same direction as idioms, more modestly.
- **Novel Metaphor** again shows no effect in the literal-pushing direction, and again
  has the most literal-skewed baseline (0.736) and highest incoherent rate at extreme
  alphas (0.088 at +3.59/+4.78 vs 0.04 at baseline), consistent with a ceiling effect
  rather than true insensitivity. On the figurative-pushing side it does move: literal
  rate falls from 0.736 (baseline) to 0.512-0.656 across alpha in [+1.2, +3.59].

Full per-category table: `analysis/summary_by_category_alpha.csv`.

## 3. Gold subset (20 expressions, one rigorously-validated variant each)

| alpha_factor | FIGURATIVE | LITERAL | INCOHERENT |
|---:|---:|---:|---:|
| -4.78 | 0.26 | 0.70 | 0.04 |
| -3.59 | 0.26 | 0.69 | 0.05 |
| -2.39 | 0.26 | 0.70 | 0.04 |
| -1.20 | 0.25 | 0.73 | 0.02 |
|  0.00 | 0.31 | 0.64 | 0.05 |
|  1.20 | 0.37 | 0.55 | 0.08 |
|  2.39 | 0.38 | 0.53 | 0.09 |
|  3.59 | 0.45 | 0.49 | 0.06 |
|  4.78 | 0.34 | 0.56 | 0.10 |

**This is the cleanest curve in the whole study.** Figurative rate rises almost
monotonically from 0.26 at alpha=-4.78 to 0.45 at alpha=+3.59 (dipping slightly at
the most extreme +4.78), with literal rate falling the mirror image. Coherence stays
>=0.90 throughout. Full table: `analysis/summary_gold_by_alpha.csv`.

## 4. Qualitative flips (majority label over 5 samples, FIGURATIVE@0 -> LITERAL@-4.78)

- **Idioms: 31/100** (idiom, variant) pairs flip, spread across 15 of the 20 idioms
  (e.g. `on ice`, `break the ice`, `in the pink`, `green light`, `up in the air`,
  `shut the door on`).
- **Figurative benchmark: 24/100** (category, expression, variant) triples flip:
  12 Conventional Metaphor, 10 Simile, 2 Novel Metaphor, a category split mirroring
  the quantitative deltas above (Simile and Conventional Metaphor generalize, Novel
  Metaphor barely does).

**Idiom example ("on ice", variant 1)**, prefix: *"Unsure if the deal would go
through, the executives kept a bottle of champagne on ice"*. Baseline continuations
read the business-deal-in-waiting sense (FIGURATIVE: "...ready to celebrate the deal
with a bottle of champagne, but they were unsure if the deal would go through").
Steered (alpha=-4.78) continuations describe an actual chilled bottle on a physical
surface (LITERAL: "...deck in case the deal fell through..."; "...hooks on the wall.
...the champagne bottle was never returned to the wall.").

**Figurative example ("like a severed cable", Simile, variant 4)**, prefix:
*"Following the CEO's abrupt resignation, the connection between headquarters and the
regional branch felt like a severed cable"*. Baseline is mixed (LITERAL/FIGURATIVE/
INCOHERENT across samples). Steered (alpha=-4.78) continuations consistently
literalize "cable/cord" as a physical wire ("...cord. The regional branch...was now
left to fend for itself..."; "...cable. Although the regional branch was still able to
maintain the core operation...").

## 5. Takeaway for the report

With both benchmarks rebuilt to the paper's full 100-item size and validated, the
idiom-derived steering vector generalizes to non-idiomatic figurative language more
convincingly than in the earlier (41-row) pass: a clean bidirectional dose-response on
the pooled figurative benchmark (comparable in cleanliness to idioms, and cleaner than
idioms on the gold subset), a strong transfer to Similes (larger delta than idioms),
a moderate transfer to Conventional Metaphors, and a flat, ceiling-limited result for
Novel Metaphors. The idiom replication itself is also far cleaner than before
(monotonic-ish, bidirectional, no more non-monotonic positive-alpha spike), though its
absolute baseline (0.436) is still well above the paper's reported ~0.12. A small
(n=86) judge-vs-human calibration (see `report/report.tex`, "Judge calibration against
human labels") gives Cohen's kappa=0.494 against our own hand labels, well below the
paper's own kappa=0.821, with a literal-leaning judge bias, making judge quality a real,
evidenced contributor to this gap alongside possible differences in how
figurative-skewed our validated prefixes are by default. Frame the conclusion as: **broader
and cleaner support for a general figurative-literal axis than the earlier pass
showed, with Simile and Conventional Metaphor generalizing most strongly, Novel
Metaphor showing a ceiling effect rather than true insensitivity, and the idiom
baseline's absolute level (not its shape) remaining the main point of divergence from
the original paper.**
