# Validation protocol — annotator prompt

This is the dual-completion validation protocol from the *Steering Idioms* (IdioSteer)
paper, Appendix ("IdiomSteer Construction and Validation Details" — Validation protocol),
adapted for our two eval benchmarks:

- `data/idiom_eval_benchmark_draft.csv` (columns: `idiom, variant_id, prefix`)
- `data/figurative_eval_benchmark.csv` (columns: `category, expression, variant_id, prefix, gold`)

Run this prompt **twice, independently** (two separate annotators, or two separate
model calls/sessions that cannot see each other's output) against the same file.
Two independent passes are required to compute inter-annotator agreement and to
match the paper's own methodology — a single pass is not a substitute.

If the annotator is a human: give them this file and the CSV, nothing else.
If the annotator is an LLM: paste everything below the `---` line as the prompt,
with the target CSV's rows substituted in, and do not show it any other annotator's
output or any prior validation notes from this project.

---

You are validating a benchmark of sentence prefixes for a figurative-vs-literal
language steering experiment. Each prefix ends exactly at a target expression
(an idiom, conventional metaphor, novel metaphor, or simile) with no text after it.
The expression is meant to be genuinely ambiguous: a reader who saw only the
prefix should be able to naturally continue it either toward the expression's
**figurative** (idiomatic/metaphorical) meaning, or toward its **literal**
(compositional, word-for-word) meaning.

You will be given a numbered list of rows, each with an expression and a prefix.

For **each row**, do the following, in order:

1. **Figurative continuation**: write one short, natural sentence-ending
   continuation (a clause or short sentence) that continues the prefix under
   the expression's figurative/idiomatic meaning.
2. **Literal continuation**: write one short, natural continuation that
   continues the prefix under the expression's literal, compositional meaning
   (i.e., treating the words as if the expression were not idiomatic at all).
3. **Verdict**: mark the row `VALID` if both continuations you just wrote are
   natural and plausible given only the prefix, and the prefix itself does not
   already tip the reader off toward one reading (e.g. it doesn't name an
   object, setting, or fact that makes the other reading nonsensical). Otherwise
   mark it `INVALID`.
   - The two readings do **not** need to be equally likely. One can be far more
     probable than the other. The bar is *plausibility*, not *balance*. Only mark
     `INVALID` if one continuation would require ignoring or contradicting
     something the prefix already established.
4. **If INVALID**, propose a concrete revision: a new prefix for the same
   expression, same general length and style, ending at the expression with no
   text after it, that would fix the problem (e.g. by changing the subject,
   verb, or scenario) while still reading naturally.

Do not try to guess which reading the author intended, and do not let the
row's position, expression frequency, or your own prior familiarity with the
expression bias your continuations — write the most natural continuation you
can in each direction, independently.

## Output format

Return one block per row, in this exact format, so two annotators' outputs can
be diffed automatically:

```
ROW <id> | <expression>
FIG: <figurative continuation>
LIT: <literal continuation>
VERDICT: VALID | INVALID
REVISION: <only if INVALID — the replacement prefix, else omit this line>
```

At the end, print one summary line:

```
SUMMARY: <n_valid>/<n_total> valid, <n_invalid> flagged
```

## After both passes

Compare the two annotators' `VERDICT` lines row by row:

- Rows both marked `VALID` → keep as-is.
- Rows both marked `INVALID` → revise using whichever proposed revision reads
  better (or synthesize one), then re-run this same protocol on the revised
  row only.
- Rows where the two annotators disagree → that disagreement is itself useful
  signal (the paper's own two-annotator process treats any flagged case as
  needing revision) — treat as `INVALID` and revise, rather than tie-breaking
  by fiat.

Report agreement as: rows where both annotators agreed (VALID/VALID or
INVALID/INVALID) ÷ total rows — this is the number to cite as the
inter-annotator agreement for this benchmark, analogous to the paper's
Cohen's κ for its own human annotators (κ = 0.867 on their 280-item gold set).
