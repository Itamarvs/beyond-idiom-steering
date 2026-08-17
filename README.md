# Beyond Idiom Steering

**Drowning in Meaning: Steering Figurative-vs-Literal Interpretation Beyond Idioms**

Final project for the NLP course (Reichman University, 2026).

## Overview

[*Steering Idioms*](https://cdn-uploads.piazza.com/paste/k262tgk1osn1jn/14bcd25454317e4a8d3df400b612d959a8156fe991aa66998fc5f4d843753129/IdioSteer.pdf) shows that residual-stream activation steering can shift LLMs between figurative and literal readings of idiomatic expressions, using a mean-difference steering vector applied cross-layer.

This project asks: **does the same steering direction reflect a general figurative-vs-literal axis**, or is its effect idiom-specific? We test this by:

1. Reproducing the idiom-steering pipeline (mean-difference vector, cross-layer additive steering, LLM-as-judge evaluation). **Done.**
2. Building a new benchmark of **non-idiomatic figurative expressions** (conventional metaphors, novel metaphors, similes) with matched figurative/literal ambiguity. **Done** (first version; may still need a validation pass).
3. Applying the idiom-derived steering vector to that benchmark and comparing figurative/literal rates, coherence, and error patterns across expression types. **In progress.**
4. *(Optional)* Deriving a second steering vector from the new figurative data and comparing its geometry (cosine similarity) to the idiom-based vector.

## Repository structure

```
beyond-idiom-steering/
├── README.md
├── requirements.txt
├── src/
│   └── steering_pipeline.py          # single shared module: model/vector loading, steering
│                                      #  hook, resumable generation + judge loops, metrics.
│                                      #  Imported by both notebooks -- no logic duplicated.
├── notebooks/                        # GPU/Colab runners, thin wrappers over src/
│   ├── step_1_build_vector_and_eval_idioms.ipynb    # builds + saves the steering vector,
│   │                                                 #  evaluates it on idioms
│   └── step_3_eval_figurative_benchmark.ipynb       # loads the saved vector, evaluates it
│                                                     #  on the figurative benchmark
├── analysis/                         # local, CPU-only -- works off committed CSVs in results/
│   └── compare_idiom_vs_figurative.py               # idiom vs. figurative tables + plot
├── data/
│   ├── idiom_eval_benchmark_draft.csv        # 20-idiom IdioSteer eval set, 5 variants/idiom (Step 1)
│   ├── figurative_eval_benchmark.csv         # 100-item metaphor/simile eval set (Step 2)
│   └── archive/
│       └── figurative_benchmark_draft.csv    # superseded first draft, kept for reference only
└── results/                          # generated artifacts (committed after each Colab run)
    ├── steering_vector_llama3.2-3b.pkl              # still valid -- built only from IdioLink,
    │                                                 #  unaffected by the benchmark rebuild below
    └── archive/                                     # superseded artifacts, kept for reference,
        ├── labeled_idiom_results_qwen3b_baseline.csv #  not read by any notebook
        ├── raw_idiom_results.csv                     #  the four below predate the 2026-08-18
        ├── labeled_idiom_results.csv                 #  benchmark rebuild (idiom_eval_benchmark_
        ├── raw_figurative_results.csv                #  draft.csv went from 41 to 100 rows,
        └── labeled_figurative_results.csv            #  figurative_eval_benchmark.csv was rebuilt
                                                       #  from scratch)
```

## Data

- **Steering vector training data**: [IdioLink](https://huggingface.co/datasets/Intellexus/IdioLink) (CC-BY-4.0), loaded directly via `datasets.load_dataset("Intellexus/IdioLink", ...)` inside `load_idiolink_pool()` -- no manual download needed. Combines the `indexes` and `queries` configs and applies the paper's exact-surface-form filter.
- **Idiom evaluation set**: `data/idiom_eval_benchmark_draft.csv` -- 20-idiom, ambiguous-prefix benchmark, 100 rows (20 idioms x 5 variants), matching IdioSteer's own benchmark size and idiom inventory exactly (Appendix, idiom list + construction constraints extracted directly from the paper PDF; the 20 idioms are `up in the air, in hot water, out of the blue, break the ice, behind bars, on ice, see red, pull someone's leg, at the crossroads, lie low, shut the door on, under the sun, take a bow, have a ball, in stitches, in the pink, green light, rough edges, in the fast lane, spill the beans`). Each prefix ends exactly at the idiom (no post-idiom text) and is written so a continuation is plausible under both a figurative and a literal reading (imbalance in likelihood is fine per the paper's own criterion -- bidirectional plausibility, not balanced probability). Single-author draft; the paper's own two-independent-annotator dual-completion validation pass hasn't been run against these rows.
- **Figurative extension set**: `data/figurative_eval_benchmark.csv` -- 100 items across 20 expressions: 50 Conventional Metaphor (`feed the fire, build a wall, open the door, light the fuse, plant the seeds, change gears, cut the ties, drop the anchor, lay the foundation, steer the ship`), 25 Novel Metaphor (`dismantle the scaffolding, overwrite the DNA, drain the aquarium, recalibrate the compass, reset the thermostat`), 25 Simile (`like a severed cable, like a rusted lock, like an overgrown garden, like a corrupted file, like an empty vault`). Rebuilt from scratch (2026-08-17): the first draft's prefixes mostly forced a single reading rather than staying ambiguous (e.g. context naming an "emotional toll" or a "trade embargo" precludes any literal continuation; conversely a named lake/coral-reef/construction-site precludes any figurative one), and a few expressions (`like a rusted lock`, `like a severed cable`, `starve the yeast`) had zero plausible reading in one direction across all 5 of their variants -- the opposite of what the IdioSteer-style design requires. Every current prefix instead pairs a concrete object/scene consistent with the literal reading (an actual fire, wall, door, cable, mechanism, etc.) with a background social/organizational tension consistent with the figurative one, per the same construction constraints as `idiom_eval_benchmark_draft.csv`: expression is sentence-final with no post-idiom text, context forces neither reading, both continuations are natural (not necessarily equally likely), and the 5 variants per expression differ in wording and scenario. Columns: `category, expression, variant_id, prefix, gold` (the column is named `expression`, not `idiom` -- these are deliberately non-idiomatic). The `gold` column flags one variant per expression (20 rows) judged the most bidirectionally balanced, for judge calibration / spot checks -- single-author judgment, not yet cross-checked by a second annotator.
- **Archived**: `data/archive/figurative_benchmark_draft.csv` -- an earlier 35-item draft (prefixes only, no gold continuations), superseded by `figurative_eval_benchmark.csv`. Kept for reference, not used by any notebook.

## Method

Following the original paper's protocol:

- **Steering vector**: mean-difference between figurative- and literal-class residual activations at a mid-to-late source layer (`v_MD = (h̄_fig − h̄_lit) / ‖h̄_fig − h̄_lit‖`), built once on IdioLink in Step 1 and reused (not rebuilt) in Step 3.
- **Steering operator**: additive (`h' = h + α·v`), applied cross-layer (vector built at source layer `Ls=14`, injected at earlier layer `Li=2`).
- **Application**: single-token intervention at the final token of the idiom/figurative expression, prompt-pass-only (no per-token steering during generation).
- **Evaluation**: LLM-as-judge (local, structured CoT prompt: state the figurative meaning, then the literal meaning, then check coherence, then decide) labels each continuation as figurative / literal / incoherent; we report literal rate and coherence rate per condition, and (Step 3) per figurative category.
- **Model**: `meta-llama/Llama-3.2-3B`, base (non-instruct), fp16.
- **Coefficient grid**: `alpha_factor ∈ {±1.20, ±2.39, ±3.59, ±4.78, 0.0}` (9 points), `n_samples=5` per condition. This exactly matches the paper's calibrated Llama-3.2-3B config (`L_s=14, L_i=2`, Table 4) -- confirmed by fetching and reading the actual paper PDF.

### What the paper actually did for judging (and why the first judge was swapped out)

The Steering Idioms paper's judge is **Gemma-4-31B-it** (Appendix J), selected by benchmarking several candidates (GPT-4o, GPT-4o-mini, Gemini 2.5 Flash, Claude Sonnet 3.5, Claude Haiku, DeepSeek, Llama-3.3-70B-Instruct, Qwen3, Gemma-4-31B-it) across ~14 prompt variants against a 280-item human-annotated gold set (two annotators, Cohen's κ=0.867 inter-annotator agreement). Gemma-4-31B-it won: 90.0% accuracy, κ=0.821 vs. gold.

Step 1's first labeling pass used a local **Qwen2.5-3B-Instruct** judge (small enough to fit alongside the 3B generation model on a free Colab T4) and produced a **flat literal rate across the entire alpha grid** (~0.30–0.41, no clear monotonic shift) -- kept for reference at `results/archive/labeled_idiom_results_qwen3b_baseline.csv`. That's a striking contrast with the paper's own reported result at this exact config (Table 6): baseline literal rate 0.12 → **0.49 at α_factor=−4.78** (only 0.13 at +4.78) -- a sharp, asymmetric shift. Since the steering mechanics (layers, alpha grid, model) are an exact match, this points at the judge as the likely weak link, not the steering vector.

Appendix Q also reveals the paper's own judge wasn't self-hosted: Gemma-4-31B-it was accessed via the **OpenRouter API** (temperature 0.0, max_tokens=700; 358,700 calls across the whole project, $91.77 total). A 31B judge doesn't fit a free-tier Colab GPU, and this project isn't adding another paid API subscription, so the judge is now **Qwen3-14B, 4-bit-quantized** (`load_judge_model(..., load_in_4bit=True)`, needs `bitsandbytes`) -- the largest local judge that reliably fits. This is a meaningful capacity step up from the 3B judge, and Qwen3 was in the paper's candidate pool (unlike Qwen2.5), but **it wasn't the winner and its standing among the candidates isn't disclosed** -- the paper only reports the winning judge's score. Treat its labels as provisional until the small human-calibration pass below is run.

`enable_thinking=False` is passed to Qwen3's chat template so it answers directly rather than emitting a `<think>...</think>` reasoning block first (the judge prompt already asks for structured reasoning steps); `judge_label()` also strips any leaked `<think>` block as a safety net in case a template revision doesn't honor the flag.

The judge prompt was also rewritten to match the paper's structure more closely (Appendix J): state the figurative meaning, then the literal meaning, then check coherence, then decide -- rather than jumping straight to a label -- including the paper's two documented judge failure modes to guard against (don't conflate a literal *physical action* with a literal reading when it realizes the idiom's own conventional frame, e.g. bowing on stage for "take a bow"; don't over-label unusual-but-coherent literal continuations as incoherent). `results/labeled_idiom_results.csv` will hold the new judge's labels once the notebook is rerun.

**Still needed regardless of judge choice**: a small human-calibration pass, mirroring the paper's methodology at a scale one person can do -- hand-label ~40 continuations sampled from `results/raw_idiom_results.csv` (mixed across alphas) and compute agreement with the judge. Not yet implemented.

## Setup

```bash
pip install -r requirements.txt
```

Run the notebooks in `notebooks/` on a GPU runtime (Colab T4 or better; both a base LLM and an instruct judge model load sequentially, not concurrently, to fit in 16GB):

1. `step_1_build_vector_and_eval_idioms.ipynb` -- loads (or, if missing, builds and saves) the idiom steering vector, evaluates it on `idiom_eval_benchmark_draft.csv` (100 rows), producing `results/raw_idiom_results.csv` + `results/labeled_idiom_results.csv`.
2. `step_3_eval_figurative_benchmark.ipynb` -- loads the vector Step 1 saved (does **not** rebuild it), first runs a cheap smoke test on just the 20 `gold` rows (`results/raw_figurative_gold_smoketest.csv`) to catch pipeline issues before the full grid, then evaluates the full benchmark, producing `results/raw_figurative_results.csv` + `results/labeled_figurative_results.csv`. Also reports a gold-subset-only breakdown alongside the full and per-category ones.

The old results generated against the pre-rebuild benchmarks (41-row idiom draft, first-pass figurative draft) are archived at `results/archive/` -- kept for reference, not read by either notebook. Both notebooks write fresh output straight to `results/` (not `results/archive/`), so just rerun them; nothing needs to be moved by hand afterward.

Then, locally (no GPU needed):

```bash
python analysis/compare_idiom_vs_figurative.py
```

produces `analysis/summary_by_alpha.csv`, `analysis/summary_by_category_alpha.csv`, `analysis/summary_gold_by_alpha.csv` (gold-subset-only, higher-confidence slice), and `analysis/literal_rate_vs_alpha.png` for the report.

## Status

- [x] Protocol extracted from paper (including the actual judge and layer/alpha config, confirmed by reading the paper PDF directly)
- [x] Idiom + figurative benchmarks rebuilt to 100 rows each and passed a dual-annotator validation protocol (see `validation/annotator_prompt.md`), converged after several revision rounds -- both bidirectionally ambiguous per-row and confirmed non-idiomatic (compositional) where relevant
- [x] Pipeline built and run: steering vector constructed on real IdioLink data (still valid -- unaffected by the benchmark rebuild, no need to rebuild it)
- [ ] Rerun Step 1's idiom eval sweep + Qwen3-14B (4-bit) judge labeling against the rebuilt 100-row benchmark and confirm the literal-rate shift now looks like the paper's -- see "What the paper actually did for judging" above. (Old results predating the rebuild are archived at `results/archive/` for reference, not read by the notebook.)
- [ ] Small human-calibration pass (~40 hand-labeled items) to get an actual agreement number for the Qwen3-14B judge, mirroring the paper's methodology at a feasible scale
- [ ] Run Step 3: full figurative-benchmark eval sweep + judge labeling against the rebuilt benchmark
- [ ] Analysis: idiom vs. figurative literal-rate comparison, by category, qualitative examples
- [ ] Optional: second steering vector from figurative data + geometry comparison (Step 4)
- [ ] Report, slides, presentation video

## Course context

Final Project, NLP 2026, Reichman University. Report max 8 pages (Overleaf template), + slides + 5-min presentation video, due 20/8/2026.
