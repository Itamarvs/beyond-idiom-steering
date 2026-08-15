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
│   ├── idiom_eval_benchmark_draft.csv        # 20-idiom IdiomSteer-style eval set (Step 1)
│   ├── figurative_eval_benchmark.csv         # 100-item metaphor/simile eval set (Step 2)
│   └── archive/
│       └── figurative_benchmark_draft.csv    # superseded first draft, kept for reference only
└── results/                          # generated artifacts (committed after each Colab run)
    ├── steering_vector_llama3.2-3b.pkl
    ├── raw_idiom_results.csv
    ├── labeled_idiom_results.csv
    ├── raw_figurative_results.csv        # produced by Step 3
    └── labeled_figurative_results.csv    # produced by Step 3
```

## Data

- **Steering vector training data**: [IdioLink](https://huggingface.co/datasets/Intellexus/IdioLink) (CC-BY-4.0), loaded directly via `datasets.load_dataset("Intellexus/IdioLink", ...)` inside `load_idiolink_pool()` -- no manual download needed. Combines the `indexes` and `queries` configs and applies the paper's exact-surface-form filter.
- **Idiom evaluation set**: `data/idiom_eval_benchmark_draft.csv` -- 20-idiom, ambiguous-prefix benchmark, 41 rows (idiom x variant).
- **Figurative extension set**: `data/figurative_eval_benchmark.csv` -- 100 items across 20 expressions: 50 Conventional Metaphor, 25 Novel Metaphor, 25 Simile, each with an ambiguous prefix plus gold `expected_literal` / `expected_figurative` continuations (used to sanity-check the judge, not fed to the model).
- **Archived**: `data/archive/figurative_benchmark_draft.csv` -- an earlier 35-item draft (prefixes only, no gold continuations), superseded by `figurative_eval_benchmark.csv`. Kept for reference, not used by any notebook.

## Method

Following the original paper's protocol:

- **Steering vector**: mean-difference between figurative- and literal-class residual activations at a mid-to-late source layer (`v_MD = (h̄_fig − h̄_lit) / ‖h̄_fig − h̄_lit‖`), built once on IdioLink in Step 1 and reused (not rebuilt) in Step 3.
- **Steering operator**: additive (`h' = h + α·v`), applied cross-layer (vector built at source layer `Ls=14`, injected at earlier layer `Li=2`).
- **Application**: single-token intervention at the final token of the idiom/figurative expression, prompt-pass-only (no per-token steering during generation).
- **Evaluation**: LLM-as-judge (Qwen2.5-3B-Instruct, local, structured CoT prompt) labels each continuation as figurative / literal / incoherent; we report literal rate and coherence rate per condition, and (Step 3) per figurative category.
- **Model**: `meta-llama/Llama-3.2-3B`, base (non-instruct), fp16.
- **Coefficient grid**: `alpha_factor ∈ {±1.20, ±2.39, ±3.59, ±4.78, 0.0}` (9 points), `n_samples=5` per condition.

### Known limitation: judge signal is currently weak

Step 1's labeled idiom results show a **flat literal rate across the entire alpha grid** (~0.30–0.41, no clear monotonic shift), unlike the sharp effect reported in the paper. Before trusting Step 3's idiom-vs-figurative comparison, this needs to be understood: either (a) the local Qwen2.5-3B-Instruct judge is too noisy to detect a real steering effect, or (b) the steering effect itself is weak in this reproduction (wrong layer, wrong scale, etc.). The sanity-check cells in `step_1_build_vector_and_eval_idioms.ipynb` (raw generations at a few alphas, read by eye) are the fastest way to tell these apart. If the judge is at fault, the fallback discussed but not yet implemented is a GPT-4o judge (prompt already drafted, commented out in the notebook history) for at least a validation subset.

## Setup

```bash
pip install -r requirements.txt
```

Run the notebooks in `notebooks/` on a GPU runtime (Colab T4 or better; both a base LLM and an instruct judge model load sequentially, not concurrently, to fit in 16GB):

1. `step_1_build_vector_and_eval_idioms.ipynb` -- builds and saves the idiom steering vector, evaluates it on `idiom_eval_benchmark_draft.csv`, produces `results/raw_idiom_results.csv` + `results/labeled_idiom_results.csv`.
2. `step_3_eval_figurative_benchmark.ipynb` -- loads the vector Step 1 saved (does **not** rebuild it), evaluates it on `figurative_eval_benchmark.csv`, produces `results/raw_figurative_results.csv` + `results/labeled_figurative_results.csv`.

Then, locally (no GPU needed):

```bash
python analysis/compare_idiom_vs_figurative.py
```

produces `analysis/summary_by_alpha.csv`, `analysis/summary_by_category_alpha.csv`, and `analysis/literal_rate_vs_alpha.png` for the report.

## Status

- [x] Protocol extracted from paper
- [x] Idiom + figurative benchmarks built (figurative benchmark may still need a validation pass)
- [x] Pipeline built and run: steering vector constructed on real IdioLink data, full idiom eval sweep + judge labeling done
- [ ] Diagnose flat literal-rate signal in Step 1 idiom results (judge vs. steering effect) -- see "Known limitation" above
- [ ] Run Step 3: full figurative-benchmark eval sweep + judge labeling
- [ ] Analysis: idiom vs. figurative literal-rate comparison, by category, qualitative examples
- [ ] Optional: second steering vector from figurative data + geometry comparison (Step 4)
- [ ] Report, slides, presentation video

## Course context

Final Project, NLP 2026, Reichman University. Report max 8 pages (Overleaf template), + slides + 5-min presentation video, due 20/8/2026.
