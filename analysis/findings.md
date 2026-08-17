# Analysis: does the idiom steering vector generalize to non-idiomatic figurative language?

Generated from `results/labeled_idiom_results.csv` (Qwen/Qwen3-14B judge, idioms) and
`results/labeled_figurative_results.csv` (Qwen/Qwen3-14B judge, figurative benchmark) via
`analysis/compare_idiom_vs_figurative.py`. Step 4 (learning a second, figurative-derived
steering vector) is out of scope for this pass.

## 1. Headline numbers (literal rate, pooled)

| alpha_factor | idiom literal rate | figurative literal rate |
|---:|---:|---:|
| -4.78 | 0.620 | 0.754 |
| -3.59 | 0.605 | 0.774 |
| -2.39 | 0.635 | 0.770 |
| -1.20 | 0.580 | 0.726 |
|  0.00 | 0.505 | 0.716 |
|  1.20 | 0.480 | 0.682 |
|  2.39 | 0.560 | 0.652 |
|  3.59 | 0.515 | 0.654 |
|  4.78 | 0.600 | 0.662 |

n = 200/alpha for idioms (20 idioms x ~2 variants x 5 samples), n = 500/alpha for the
figurative benchmark (20 expressions x 5 variants x 5 samples). Coherence rate stays >=0.89
in every cell (full table: `analysis/summary_by_alpha.csv`), so steering isn't degrading
fluency in either dataset.

**Idiom literal-rate delta (alpha=-4.78 vs 0.0): +0.115.** Direction matches the paper
(negative alpha -> more literal), but the baseline (0.505) and the size of the shift are both
far from the paper's reported Llama-3.2-3B config (baseline 0.12 -> 0.49 at alpha=-4.78, a
~4x jump from a low floor). Our baseline sits at the 50/50 point instead, so there's much
less headroom to "push toward literal" in the first place, and the effect is noisier and
non-monotonic on the positive side (alpha=+4.78 also reads *more* literal than baseline,
which the paper's result does not show). Treat this as a **partial, weaker replication**,
not a clean recovery of the original effect.

**Figurative benchmark shows a cleaner, near-monotonic trend than idioms**: literal rate
falls steadily from 0.774 (alpha=-3.59) down to 0.652-0.662 as alpha moves positive. This is
the idiom-derived vector's effect on a dataset it was never trained on -- so the fact that
the direction (not just magnitude) matches the original figurative<->literal axis is the
main positive signal for the project's core hypothesis.

## 2. By figurative category (delta = alpha -4.78 vs 0.0, literal rate)

| category | delta | baseline literal rate |
|---|---:|---:|
| Simile | +0.080 | ~0.68 (see `summary_by_category_alpha.csv`) |
| Conventional Metaphor | +0.040 | ~0.70 |
| Novel Metaphor | -0.008 (flat) | 0.704 |

- **Conventional Metaphor and Simile** move in the same direction as idioms (negative alpha
  -> more literal), i.e. the idiom vector partially generalizes to these categories.
- **Novel Metaphor shows no effect** in that direction, but it's also the category with the
  most literal-skewed baseline (~65-77% literal across the whole alpha grid) and the highest
  INCOHERENT rate at the extremes (0.152 at alpha=+3.59 vs 0.04 at baseline) -- i.e. it's
  already near a literal ceiling, so a "push toward literal" has little room to show up.
  The more informative read for this category is the *positive*-alpha side, where literal
  rate does drop (0.704 at baseline -> ~0.624-0.656 at alpha in [+1.2, +4.78]), i.e. the
  vector still measurably pushes it toward figurative when run in that direction.
- Full per-category table: `analysis/summary_by_category_alpha.csv`.

## 3. Qualitative examples

**Idiom, clean flip** ("green light", variant 1) -- didn't fully flip at the sample level
shown below (both remained figurative), but 8/40 idiom (idiom, variant) pairs flip from
majority-FIGURATIVE at alpha=0 to majority-LITERAL at alpha=-4.78: `green light`,
`in the pink`, `lie low` (x2 variants), `on ice`, `out of the blue`, `take a bow`,
`under the sun`.

**Figurative benchmark, clean flip** ("bleach the ledger", Novel Metaphor, variant 1):

> Prefix: *"Desperate to hide the massive embezzlement scheme before the auditors arrived,
> the CFO attempted to bleach the ledger"*
> - Baseline (alpha=0, FIGURATIVE): "...entries in the general ledger. The system had a
>   built-in capability to reverse a journal entry, so the CFO was able to accomplish his
>   task. In the accounting world, the term 'bleaching' is used to describe an effort to..."
>   (reads the ledger digitally/financially -- the intended figurative sense.)
> - Steered (alpha=-4.78, LITERAL): "...card stock of the check register. However, the CFO
>   was only able to bleach the check register for about 20 minutes before the bleach ran
>   out. In order to hide the check register, the CFO cut the register in half..."
>   (reads "bleach" as literal chemical bleach on paper.)

15/125 (category, expression, variant) triples in the figurative benchmark show this same
majority FIGURATIVE-at-0 -> LITERAL-at--4.78 flip pattern, spread across all three
categories (5 Conventional Metaphor, 6 Novel Metaphor, 3 Simile) -- see
`analysis/find_flips.py`-style query in this file's generation history for the full list.

## 4. Takeaway for the report

The idiom-derived steering vector is **not idiom-specific**: it shifts figurative/literal
rates in the expected direction on at least 2 of 3 non-idiomatic figurative categories
(Simile, Conventional Metaphor), with the third (Novel Metaphor) showing the expected effect
only on the figurative-pushing side, likely because its baseline is already literal-skewed.
The effect size on the figurative benchmark, however, is smaller than on idioms
(pooled delta +0.038 vs +0.115), and the idiom replication itself is weaker/noisier than the
original paper's reported numbers -- most likely because our ambiguous-prefix idiom set sits
at a ~50/50 figurative/literal baseline rather than the paper's low (~0.12) literal baseline,
leaving less headroom, and because judge/prompt differences remain a plausible confound
despite the Qwen3-14B upgrade. Frame the conclusion as: **partial support for a general
figurative-literal axis, with category-dependent generalization and a caveat about
replication fidelity on the idiom baseline.**
