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

# Judge model. The Steering Idioms paper (Appendix J) selected Gemma-4-31B-it
# after benchmarking several candidates (incl. GPT-4o, Claude Sonnet 3.5,
# Llama-3.3-70B-Instruct, Qwen3) against a 280-item human-annotated gold set
# (90.0% accuracy, Cohen's kappa=0.821 vs. gold; human-human kappa=0.867) --
# accessed via the OpenRouter API, not self-hosted (Appendix Q). Only the
# winning judge's score is reported; Qwen3's standing among the candidates
# is not disclosed. A 31B judge doesn't fit alongside the 3B generation
# model on a free-tier Colab GPU, so this uses Qwen3-14B (4-bit, the
# largest judge that reliably fits) as an UNVALIDATED substitute -- it is
# in the paper's candidate pool (so evidence-adjacent) but was not the
# winner and has no calibration evidence of its own here. Treat its output
# as provisional until the small human-calibration pass (see README) is
# run. This replaced the original Qwen2.5-3B-Instruct judge, which produced
# a flat, unreliable literal-rate signal.
JUDGE_MODEL_NAME = "Qwen/Qwen3-14B"

# Structured-reasoning prompt mirroring the paper's judge design (Appendix
# J): state the figurative meaning, then the literal meaning, then check
# coherence, then decide -- rather than jumping straight to a label. Guards
# against the paper's two documented judge failure modes: (1) don't conflate
# a literal *physical action* with a literal reading when that action
# realizes the idiom's own conventional frame (e.g. bowing on stage for
# "take a bow" is still FIGURATIVE); (2) don't over-label unusual-but-
# coherent literal continuations as INCOHERENT.
JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a continuation of a sentence ending in the expression "{expression}" interprets the expression FIGURATIVELY, LITERALLY, or is INCOHERENT.

Sentence prefix: "{prefix}"
Continuation: "{continuation}"

Step 1: State the conventional, non-compositional figurative meaning of "{expression}".
Step 2: State the literal, compositional meaning of "{expression}"'s component words.
Step 3: Check coherence: is the continuation syntactically well-formed and does it describe an imaginable situation (treating the expression as ordinary words if needed)? If not, the label is INCOHERENT.
Step 4: If coherent, decide whether the continuation is clearly better supported by the figurative or literal reading.
  - Physical actions that realize the conventional figurative frame (e.g. bowing on stage after a play, for "take a bow") are still FIGURATIVE.
  - Only label LITERAL if the component words are used compositionally, outside the figurative frame.
  - Default to FIGURATIVE if genuinely ambiguous.
Step 5: Output the final label formatted exactly as: FINAL LABEL: <FIGURATIVE/LITERAL/INCOHERENT>"""


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


def load_judge_model(judge_model_name: str = JUDGE_MODEL_NAME, device: Optional[str] = None,
                      load_in_4bit: bool = True):
    """Loads the instruct LLM used for figurative/literal/incoherent labeling.

    load_in_4bit=True (default) quantizes via bitsandbytes -- required to fit
    a 14B-class judge on a free-tier Colab GPU. Set False for smaller judges
    (e.g. the original Qwen2.5-3B-Instruct) where fp16 fits comfortably."""
    device = device or get_device()
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
    # Left-padding is required for batched causal-LM generation (so every
    # sequence's "next token to generate" lines up at the same position).
    judge_tokenizer.padding_side = "left"
    if judge_tokenizer.pad_token is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
        )
        judge_model = AutoModelForCausalLM.from_pretrained(
            judge_model_name, quantization_config=quant_config, device_map=device
        )
    else:
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
    return [tokenizer.decode(seq[prompt_len:], skip_special_tokens=True,
                              clean_up_tokenization_spaces=False) for seq in out_ids]


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

def _parse_judge_response(response_text: str) -> str:
    # Safety net in case enable_thinking=False isn't fully honored: strip any
    # leaked <think>...</think> block before looking for the final label.
    response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip().upper()
    match = re.search(r"FINAL\s*LABEL\s*:\s*(FIGURATIVE|LITERAL|INCOHERENT)", response_text)
    if match:
        return match.group(1)
    for label in ["INCOHERENT", "LITERAL", "FIGURATIVE"]:
        if label in response_text:
            return label
    return "UNPARSEABLE"


def judge_label(judge_model, judge_tokenizer, device, expression: str, prefix: str, continuation: str) -> str:
    """Labels a single continuation. For bulk labeling use judge_label_batch /
    run_judge_eval instead -- one-at-a-time generate() calls are far slower
    per item on a GPU than a batched call (T4 + 4-bit 14B: ~25-30s/item
    unbatched vs. roughly batch_size-x faster batched, since decode is
    memory-bandwidth-bound and a batch amortizes the weight-load cost)."""
    return judge_label_batch(judge_model, judge_tokenizer, device, [(expression, prefix, continuation)])[0]


def judge_label_batch(judge_model, judge_tokenizer, device,
                       triples: Sequence[tuple]) -> list:
    """Labels a batch of (expression, prefix, continuation) triples in one
    forward/generate call. Returns labels in the same order. Requires
    judge_tokenizer.padding_side == "left" (set by load_judge_model)."""
    texts = []
    for expression, prefix, continuation in triples:
        prompt = JUDGE_PROMPT_TEMPLATE.format(expression=expression, prefix=prefix, continuation=continuation)
        text = judge_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False,  # Qwen3: answer directly, no <think> preamble --
                                     # ignored harmlessly by chat templates that don't define it
        )
        texts.append(text)

    inputs = judge_tokenizer(texts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out_ids = judge_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=judge_tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    labels = []
    for seq in out_ids:
        response_text = judge_tokenizer.decode(seq[prompt_len:], skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False).strip()
        labels.append(_parse_judge_response(response_text))
    return labels


def run_judge_eval(judge_model, judge_tokenizer, device, in_csv: str, out_csv: str,
                    key_cols: Sequence[str] = ("idiom", "variant_id", "alpha_factor", "sample_i"),
                    expr_col: str = "idiom", batch_size: int = 4,
                    checkpoint_every: int = 1) -> pd.DataFrame:
    """Resumable, checkpointed labeling loop mirroring run_generation_eval.

    Two independent speed/safety levers:
      - batch_size: judge calls grouped per generate() call. Decode is
        memory-bandwidth-bound, so a bigger batch is close to free throughput
        -- unbatched labeling is ~25-30s/item on a T4 with the 4-bit 14B
        judge (35 hours for ~4,500 items); batching is what makes a full
        sweep fit a session. Raise until you see an OOM, then back off.
      - checkpoint_every: how many *batches* to accumulate in memory before
        writing+syncing to disk (default 1 = write after every batch). Raise
        this to cut disk/Drive-sync overhead if you don't need near-real-time
        visibility into progress -- a crash loses at most
        checkpoint_every * batch_size in-flight items; everything already
        written is safe, and a rerun only redoes what wasn't written.

    Each checkpoint write opens, appends, and closes the file (rather than
    holding one handle open for the whole run) -- on local disk this is
    negligible overhead; on a Google-Drive-mounted out_csv (via Colab's FUSE
    layer), an actively-open handle can leave writes buffered and invisible
    in Drive until it finally closes, so closing per checkpoint is what
    actually forces a sync, not flush() alone.

    `expr_col` names the column holding the target expression (an idiom for
    the idiom benchmark, a metaphor/simile for the figurative benchmark --
    pass expr_col="expression" for the latter)."""
    results_df = pd.read_csv(in_csv)

    done_keys = set()
    file_exists = os.path.isfile(out_csv)
    if file_exists:
        existing = pd.read_csv(out_csv)
        done_keys = set(zip(*[existing[c] for c in key_cols]))
        print(f"Resuming: {len(done_keys)} rows already labeled.")
    else:
        print("No existing labels file -- starting fresh.")

    pending = [row for _, row in results_df.iterrows()
               if tuple(row[c] for c in key_cols) not in done_keys]

    fieldnames = list(results_df.columns) + ["label"]

    if not file_exists:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    pbar = tqdm(total=len(results_df), desc="judging")
    pbar.update(len(done_keys))

    buffer = []  # (row, label) pairs accumulated since the last checkpoint
    batches_since_checkpoint = 0

    def write_checkpoint():
        if not buffer:
            return
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for buffered_row, buffered_label in buffer:
                writer.writerow({**buffered_row.to_dict(), "label": buffered_label})
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass  # not all filesystems (e.g. some FUSE mounts) support fsync
        buffer.clear()

    for i in range(0, len(pending), batch_size):
        batch_rows = pending[i:i + batch_size]
        triples = [(row[expr_col], row["prefix"], row["continuation"]) for row in batch_rows]
        labels = judge_label_batch(judge_model, judge_tokenizer, device, triples)

        buffer.extend(zip(batch_rows, labels))
        batches_since_checkpoint += 1
        pbar.update(len(batch_rows))

        if batches_since_checkpoint >= checkpoint_every:
            write_checkpoint()
            batches_since_checkpoint = 0

    write_checkpoint()  # flush any remainder below a full checkpoint interval
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
