# Beyond Idiom Steering

**Drowning in Meaning: Steering Figurative-vs-Literal Interpretation Beyond Idioms**

Final project for the NLP course (Reichman University, 2026).

## Overview

[*Steering Idioms*](https://cdn-uploads.piazza.com/paste/k262tgk1osn1jn/14bcd25454317e4a8d3df400b612d959a8156fe991aa66998fc5f4d843753129/IdioSteer.pdf) shows that residual-stream activation steering can shift LLMs between figurative and literal readings of idiomatic expressions, using a mean-difference steering vector applied cross-layer.

This project asks: **does the same steering direction reflect a general figurative-vs-literal axis**, or is its effect idiom-specific? We test this by:

1. Reproducing the idiom-steering pipeline (mean-difference vector, cross-layer additive steering, LLM-as-judge evaluation).
2. Applying the idiom-derived steering vector to a new benchmark of **non-idiomatic figurative expressions** (metaphors and similes) with matched figurative/literal ambiguity.
3. Comparing figurative/literal rates, coherence, and error patterns across expression types.
4. (Optional) Deriving a second steering vector from the new figurative data and comparing its geometry (cosine similarity) to the idiom-based vector.

## Repository structure

```
beyond-idiom-steering/
├── README.md
├── data/
│   ├── idiom_eval_benchmark_draft.csv       # reconstructed IdiomSteer-style eval set (20 idioms)
│   └── figurative_benchmark_draft.csv       # new metaphor/simile benchmark (50 items)
├── src/
│   └── steering_pipeline.py                 # vector construction, steering operator, generation, judge prompt
├── notebooks/
│   └── colab_main.ipynb                     # GPU runtime: builds vector, runs eval sweeps
├── results/                                 # generated continuations + judge labels (CSV)
└── report/                                  # Overleaf export, slides, presentation video
```

## Data

- **Steering vector training data**: [IdioLink](https://huggingface.co/datasets/Intellexus/IdioLink) (CC-BY-4.0), filtered to the 22 idiom-disjoint training idioms used in the Steering Idioms paper. Loaded directly via `datasets.load_dataset("Intellexus/IdioLink")` — no manual download needed.
- **Idiom evaluation set**: `data/idiom_eval_benchmark_draft.csv` — draft reconstruction of the paper's 20-idiom IdiomSteer-style benchmark (ambiguous prefixes ending at the idiom). Needs a second-annotator validation pass before final use.
- **Figurative extension set**: `data/figurative_benchmark_draft.csv` — 30 metaphors + 20 similes, same ambiguous-prefix design. Also needs validation.

## Method

Following the original paper's protocol:

- **Steering vector**: mean-difference between figurative- and literal-class residual activations at a mid-to-late source layer (`v_MD = (h̄_fig − h̄_lit) / ‖h̄_fig − h̄_lit‖`).
- **Steering operator**: additive (`h' = h + α·v`), applied cross-layer (vector built at source layer `Ls`, injected at earlier layer `Li`).
- **Application**: single-token intervention at the final token of the idiom/figurative expression, prompt-pass-only (no per-token steering during generation).
- **Evaluation**: LLM-as-judge labels each continuation as figurative / literal / incoherent; report literal rate and coherence rate per condition.
- **Model(s)**: base (non-instruct) open-weights LLM, starting with Llama-3.2-3B or Gemma-2-2B for compute reasons.

## Setup

```bash
pip install transformers torch accelerate datasets pandas
```

Run `notebooks/colab_main.ipynb` on a GPU runtime (Colab T4 or Kaggle Notebook). It:
1. Loads IdioLink and builds the idiom-based steering vector.
2. Runs steered/unsteered generation on both benchmark CSVs.
3. Saves raw continuations to `results/`.
4. Labels continuations with an LLM judge and computes literal/coherence rates.

## Status

- [x] Protocol extracted from paper
- [x] Draft idiom + figurative benchmarks (need validation)
- [x] Pipeline skeleton (`src/steering_pipeline.py`)
- [ ] Build steering vector on real IdioLink data
- [ ] Sanity-check steering effect on a few idiom examples
- [ ] Validate both benchmarks with a second annotator
- [ ] Full evaluation sweep (idioms + figurative extension)
- [ ] Optional: second steering vector from figurative data + geometry comparison
- [ ] Report, slides, presentation video

## Course context

Final Project, NLP 2026, Reichman University. Report max 8 pages (Overleaf template), + slides + 5-min presentation video, due 20/8/2026.
