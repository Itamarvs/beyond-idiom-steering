"""
Shared pipeline for Steering Idioms replication + figurative-benchmark extension.

Implements the protocol from "Steering Idioms: Controlling Figurative-vs-Literal
Interpretation via Residual-Stream Activation Steering": a mean-difference
steering vector, additive operator, cross-layer injection, single-token
(prompt-forward-pass-only) intervention, plus an LLM-as-judge labeling step.

This module holds everything that is identical across steps. Step-specific
notebooks (data loading for a particular benchmark, which cells to run, in
what order) live in notebooks/; this file is imported, not copy-pasted.

Used by:
  - notebooks/step_1_build_vector_and_eval_idioms.ipynb
      Builds the idiom-derived steering vector from IdioLink and evaluates it
      on data/idiom_eval_benchmark_draft.csv.
  - notebooks/step_3_eval_figurative_benchmark.ipynb
      Loads the vector saved by Step 1 and evaluates it on
      data/figurative_eval_benchmark.csv (no rebuilding).

Run on a GPU runtime (Colab / local) with: pip install -r requirements.txt
"""

import csv
import os
import pickle
import re
import time
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config shared across all steps. Keep these fixed once Step 1 has produced
# a saved vector -- Step 3 asserts the loaded vector's model matches MODEL_NAME.
# ---------------------------------------------------------------------------

MODEL_NAME = "meta-llama/Llama-3.2-3B"   # base model, not instruct (per paper design)
SOURCE_LAYER = 14                         # Ls for Llama-3.2-3B (paper Table 4)
INJECTION_LAYER = 2                       # Li, fixed across all models per paper protocol
ALPHA_FACTOR_GRID = [-4.78, -3.59, -2.39, -1.20, 0.0, 1.20, 2.39, 3.59, 4.78]
N_SAMPLES = 5
MAX_NEW_TOKENS = 50

JUDGE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a continuation of a sentence ending in the expression "{idiom}" interprets the expression FIGURATIVELY, LITERALLY, or is INCOHERENT.

Sentence prefix: "{prefix}"
Continuation: "{continuation}"

Definitions:
- FIGURATIVE: The expression is used in its conventional, non-compositional sense. Note: Physical actions that realize the conventional frame (e.g. bowing on stage for 'take a bow') are still FIGURATIVE.
- LITERAL: The component words are used compositionally outside the idiomatic/figurative frame (e.g. ice cubes, beans in a can).
- INCOHERENT: The continuation is grammatically broken, nonsensical, or fails to form an imaginable scenario.

Step 1: Briefly state if the continuation is coherent.
Step 2: State whether the reading is literal or figurative. If ambiguous, default to FIGURATIVE.
Step 3: Output the final label formatted exactly as: FINAL LABEL: <FIGURATIVE/LITERAL/INCOHERENT>"""


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_generation_model(model_name: str = MODEL_NAME, device: Optional[str] = None):
    """Loads the base (non-instruct) LLM used for steered generation."""
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device
    )
    model.eval()
    return model, tokenizer, device


def load_judge_model(judge_model_name: str = JUDGE_MODEL_NAME, device: Optional[str] = None):
    """Loads the instruct LLM used for figurative/literal/incoherent labeling."""
    device = device or get_device()
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_model_name, torch_dtype=torch.float16, device_map=device
    )
    judge_model.eval()
    return judge_model, judge_tokenizer, device


def get_layer_module(model, layer_idx: int):
    return model.model.layers[layer_idx]  # adjust path if using a non-Llama architecture


# ---------------------------------------------------------------------------
# Activation collection: residual stream at a fixed source layer, at the
# final token of the target expression (idiom span or figurative expression).
# ---------------------------------------------------------------------------

def find_span_end_token(sentence: str, span: str, tokenizer) -> Optional[int]:
    """Index of the last token whose character range covers the end of `span`
    within `sentence`. Returns None if `span` is not found."""
    start_char = sentence.lower().find(span.lower())
    if start_char == -1:
        return None
    end_char = start_char + len(span)
    enc = tokenizer(sentence, return_offsets_mapping=True, return_tensors=None)
    offsets = enc["offset_mapping"]
    end_token_idx = None
    for i, (s, e) in enumerate(offsets):
        if s < end_char <= e:
            end_token_idx = i
            break
        if s < end_char and e <= end_char:
            end_token_idx = i  # keep updating; last token whose span ends <= end_char
    return end_token_idx


def collect_activation(model, tokenizer, device, sentence: str, span: str, layer_idx: int):
    """Runs a forward pass and returns the residual-stream activation (as a
    numpy vector) at the final token of `span` within `sentence`."""
    captured = {}

    def hook(module, inp, out):
        captured["act"] = out[0] if isinstance(out, tuple) else out

    tok_idx = find_span_end_token(sentence, span, tokenizer)
    if tok_idx is None:
        return None

    handle = get_layer_module(model, layer_idx).register_forward_hook(hook)
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return captured["act"][0, tok_idx, :].float().cpu().numpy()


# ---------------------------------------------------------------------------
# IdioLink loading (Step 1 only -- the pool the idiom steering vector is
# built from). Kept here so Step 1's notebook stays a thin runner too.
# ---------------------------------------------------------------------------

def load_idiolink_pool() -> pd.DataFrame:
    """Loads IdioLink (indexes + queries splits), keeps idiomatic/literal rows
    whose span is an exact surface-form match of the idiom, per the paper's
    filtering step. Returns a DataFrame with columns: sentence, idiom, span,
    label ('figurative'/'literal')."""
    from datasets import load_dataset

    idiolink_idx = load_dataset("Intellexus/IdioLink", "indexes")
    idiolink_qry = load_dataset("Intellexus/IdioLink", "queries")

    idx_train = idiolink_idx["train"].to_pandas()
    qry_train = idiolink_qry["train"].to_pandas()

    idx_pool = idx_train[idx_train["usage"].isin(["idiomatic", "literal"])].copy()
    qry_pool = qry_train[qry_train["usage"].isin(["idiomatic", "literal"])].copy()

    pool_df = pd.concat([idx_pool, qry_pool], ignore_index=True)
    pool_df["label"] = pool_df["usage"].map({"idiomatic": "figurative", "literal": "literal"})

    # Paper's filtering step: keep only rows where the idiom's surface form is
    # literally present in the sentence (span == idiom, no inflected forms).
    pool_df = pool_df[pool_df["span"].str.lower() == pool_df["idiom"].str.lower()]
    return pool_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Mean-difference steering vector construction (Eq. 1 in the paper)
# ---------------------------------------------------------------------------

def build_steering_vector(pool_df: pd.DataFrame, model, tokenizer, device,
                           source_layer: int = SOURCE_LAYER):
    """pool_df must have columns: sentence, span, label ('figurative'/'literal')."""
    fig_acts, lit_acts = [], []
    for _, row in tqdm(pool_df.iterrows(), total=len(pool_df), desc="collecting activations"):
        act = collect_activation(model, tokenizer, device, row["sentence"], row["span"], source_layer)
        if act is None:
            continue
        (fig_acts if row["label"] == "figurative" else lit_acts).append(act)

    h_fig = np.mean(np.stack(fig_acts), axis=0)
    h_lit = np.mean(np.stack(lit_acts), axis=0)
    diff = h_fig - h_lit
    s_md = np.linalg.norm(diff)
    v_md = diff / s_md
    return v_md, s_md


def save_steering_vector(path: str, v_md: np.ndarray, s_md: float,
                          source_layer: int = SOURCE_LAYER, model_name: str = MODEL_NAME):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {"v_md": v_md, "s_md": s_md, "source_layer": source_layer, "model": model_name}, f
        )


def load_steering_vector(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Additive steering operator with cross-layer injection, single-position,
# prompt-pass-only intervention (Eq. 3, Appendix G)
# ---------------------------------------------------------------------------

def make_steering_hook(v_md_t: torch.Tensor, alpha: float, target_position: int):
    def hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        seq_len = hidden.shape[1]
        if target_position < seq_len:  # only fires on the prompt's forward pass
            hidden[:, target_position, :] = hidden[:, target_position, :] + alpha * v_md_t
        if isinstance(out, tuple):
            return (hidden,) + out[1:]
        return hidden
    return hook


def steered_generate_batch(model, tokenizer, device, prefix: str, v_md: np.ndarray, s_md: float,
                            alpha_factor: float, injection_layer: int = INJECTION_LAYER,
                            n_samples: int = N_SAMPLES, max_new_tokens: int = MAX_NEW_TOKENS):
    """Returns a list of n_samples continuations for one (prefix, alpha_factor)
    condition, generated in a single batched call."""
    inputs = tokenizer(prefix, return_tensors="pt").to(device)
    target_position = inputs["input_ids"].shape[1] - 1  # final token of the prefix

    alpha = alpha_factor * s_md
    v_md_t = torch.tensor(v_md, dtype=torch.float16, device=device)

    handle = None
    if alpha_factor != 0:
        handle = get_layer_module(model, injection_layer).register_forward_hook(
            make_steering_hook(v_md_t, alpha, target_position)
        )
    try:
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, do_sample=True, temperature=0.7, top_p=0.9,
                max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=n_samples,
            )
    finally:
        if handle is not None:
            handle.remove()

    prompt_len = inputs["input_ids"].shape[1]
    return [tokenizer.decode(seq[prompt_len:], skip_special_tokens=True) for seq in out_ids]


# ---------------------------------------------------------------------------
# Resumable, checkpointed generation loop over a benchmark CSV. Writes rows
# incrementally so a Colab disconnect only costs the current condition, and
# skips (idiom, variant_id, alpha_factor) combos already present in out_csv
# on restart.
# ---------------------------------------------------------------------------

def run_generation_eval(model, tokenizer, device, eval_df: pd.DataFrame, v_md: np.ndarray, s_md: float,
                         out_csv: str, alpha_grid: Sequence[float] = ALPHA_FACTOR_GRID,
                         n_samples: int = N_SAMPLES, injection_layer: int = INJECTION_LAYER,
                         max_new_tokens: int = MAX_NEW_TOKENS,
                         key_cols: Sequence[str] = ("idiom", "variant_id")) -> pd.DataFrame:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    done_keys = set()
    file_exists = os.path.isfile(out_csv)
    if file_exists:
        existing = pd.read_csv(out_csv)
        done_keys = set(zip(*[existing[c] for c in key_cols], existing["alpha_factor"]))
        print(f"Resuming: {len(done_keys)} conditions already done.")
    else:
        print("No existing results file -- starting fresh.")

    fieldnames = list(eval_df.columns) + ["alpha_factor", "sample_i", "continuation"]
    write_header = not file_exists

    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        total_conditions = len(eval_df) * len(alpha_grid)
        pbar = tqdm(total=total_conditions, desc="conditions")
        pbar.update(len(done_keys))

        for _, row in eval_df.iterrows():
            for alpha_factor in alpha_grid:
                key = (*[row[c] for c in key_cols], alpha_factor)
                if key in done_keys:
                    continue

                t0 = time.time()
                continuations = steered_generate_batch(
                    model, tokenizer, device, row["prefix"], v_md, s_md, alpha_factor,
                    injection_layer=injection_layer, n_samples=n_samples, max_new_tokens=max_new_tokens,
                )
                elapsed = time.time() - t0

                for sample_i, cont in enumerate(continuations):
                    out_row = {**row.to_dict(), "alpha_factor": alpha_factor,
                               "sample_i": sample_i, "continuation": cont}
                    writer.writerow(out_row)
                f.flush()

                pbar.update(1)
                pbar.set_postfix({"alpha": alpha_factor, "sec": f"{elapsed:.1f}"})

        pbar.close()

    print("Done. Full results in", out_csv)
    return pd.read_csv(out_csv)


# ---------------------------------------------------------------------------
# LLM-as-judge labeling (figurative / literal / incoherent)
# ---------------------------------------------------------------------------

def judge_label(judge_model, judge_tokenizer, device, idiom: str, prefix: str, continuation: str) -> str:
    prompt = JUDGE_PROMPT_TEMPLATE.format(idiom=idiom, prefix=prefix, continuation=continuation)
    messages = [{"role": "user", "content": prompt}]

    inputs = judge_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(device)

    with torch.no_grad():
        out_ids = judge_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=judge_tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    response_text = judge_tokenizer.decode(
        out_ids[0][prompt_len:], skip_special_tokens=True
    ).strip().upper()

    match = re.search(r"FINAL\s*LABEL\s*:\s*(FIGURATIVE|LITERAL|INCOHERENT)", response_text)
    if match:
        return match.group(1)
    for label in ["INCOHERENT", "LITERAL", "FIGURATIVE"]:
        if label in response_text:
            return label
    return "UNPARSEABLE"


def run_judge_eval(judge_model, judge_tokenizer, device, in_csv: str, out_csv: str,
                    key_cols: Sequence[str] = ("idiom", "variant_id", "alpha_factor", "sample_i")) -> pd.DataFrame:
    """Resumable, checkpointed labeling loop mirroring run_generation_eval."""
    results_df = pd.read_csv(in_csv)

    done_keys = set()
    file_exists = os.path.isfile(out_csv)
    if file_exists:
        existing = pd.read_csv(out_csv)
        done_keys = set(zip(*[existing[c] for c in key_cols]))
        print(f"Resuming: {len(done_keys)} rows already labeled.")
    else:
        print("No existing labels file -- starting fresh.")

    fieldnames = list(results_df.columns) + ["label"]
    write_header = not file_exists

    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        pbar = tqdm(total=len(results_df), desc="judging")
        pbar.update(len(done_keys))

        for _, row in results_df.iterrows():
            key = tuple(row[c] for c in key_cols)
            if key in done_keys:
                continue

            label = judge_label(judge_model, judge_tokenizer, device,
                                 row["idiom"], row["prefix"], row["continuation"])
            out_row = {**row.to_dict(), "label": label}
            writer.writerow(out_row)
            f.flush()
            pbar.update(1)

        pbar.close()

    print("Done. Labeled results in", out_csv)
    return pd.read_csv(out_csv)


# ---------------------------------------------------------------------------
# Summary metrics: figurative / literal / incoherent rate per group
# ---------------------------------------------------------------------------

def summarize_labels(labeled_df: pd.DataFrame, group_cols: Sequence[str] = ("alpha_factor",)) -> pd.DataFrame:
    group_cols = list(group_cols)
    summary = labeled_df.groupby(group_cols)["label"].value_counts(normalize=True).unstack().fillna(0)
    for col in ["FIGURATIVE", "LITERAL", "INCOHERENT"]:
        if col not in summary.columns:
            summary[col] = 0
    summary["coherence_rate"] = summary["FIGURATIVE"] + summary["LITERAL"]
    return summary[["FIGURATIVE", "LITERAL", "INCOHERENT", "coherence_rate"]].round(3)
