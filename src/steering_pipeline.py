"""
Steering Idioms replication + Figurative-Extension pipeline.
Implements the protocol from "Steering Idioms: Controlling Figurative-vs-Literal
Interpretation via Residual-Stream Activation Steering" (mean-difference vector,
additive operator, cross-layer injection, single-token intervention).

Run this on a GPU machine (Colab / local) with:
  pip install transformers torch accelerate pandas

Usage:
  1. Fill in IDIOLINK_PATH with the IdioLink train split (Intellexus/IdioLink on HF)
     to build the steering vector (Step 1.4).
  2. Point EVAL_CSV at idiom_eval_benchmark_draft.csv or figurative_benchmark_draft.csv
     (Step 1.5 / Step 3).
  3. Run build_steering_vector() once, then run_eval() for each benchmark.
"""

import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-3B"   # base model, not instruct (per paper: RQ design)
SOURCE_LAYER = 14                         # Ls, mid-to-late layer (paper: 14 for Llama-3.2-3B)
INJECTION_LAYER = 2                       # Li, fixed across all models per paper protocol
ALPHA_FACTOR_GRID = [-4.78, -3.59, -2.39, -1.20, 1.20, 2.39, 3.59, 4.78]

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device
)
model.eval()

# ---------------------------------------------------------------------------
# 1. Activation collection at a fixed source layer, final-token-of-expression
# ---------------------------------------------------------------------------

_captured = {}

def _capture_hook(layer_idx):
    def hook(module, inp, out):
        # residual stream output of this decoder layer
        _captured[layer_idx] = out[0] if isinstance(out, tuple) else out
    return hook

def get_layer_module(layer_idx):
    # adjust path if using a different architecture (Gemma vs Llama)
    return model.model.layers[layer_idx]

def collect_activation(sentence: str, layer_idx: int, target_substring: str):
    """Runs a forward pass and returns the residual-stream activation at the
    final token of `target_substring` (e.g. the idiom or figurative expression)."""
    handle = get_layer_module(layer_idx).register_forward_hook(_capture_hook(layer_idx))
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    # final token position = last token of the prompt (per paper: sentences end at expression)
    act = _captured[layer_idx][0, -1, :].float().cpu().numpy()
    return act

# ---------------------------------------------------------------------------
# 2. Mean-difference steering vector construction (Eq. 1 in the paper)
# ---------------------------------------------------------------------------

def build_steering_vector(idiolink_df: pd.DataFrame, source_layer: int = SOURCE_LAYER):
    """idiolink_df must have columns: sentence, label ('figurative'/'literal')."""
    fig_acts, lit_acts = [], []
    for _, row in idiolink_df.iterrows():
        act = collect_activation(row["sentence"], source_layer, row.get("idiom", ""))
        (fig_acts if row["label"] == "figurative" else lit_acts).append(act)
    h_fig = np.mean(fig_acts, axis=0)
    h_lit = np.mean(lit_acts, axis=0)
    diff = h_fig - h_lit
    s_md = np.linalg.norm(diff)
    v_md = diff / s_md
    return v_md, s_md  # unit vector + raw norm (for alpha_factor scaling)

# ---------------------------------------------------------------------------
# 3. Additive steering operator with cross-layer injection + single-position,
#    prompt-pass-only intervention (Eq. 3, Appendix G)
# ---------------------------------------------------------------------------

def make_steering_hook(v_md: np.ndarray, alpha: float, target_position: int):
    v_t = torch.tensor(v_md, dtype=torch.bfloat16, device=device)

    def hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        seq_len = hidden.shape[1]
        if target_position < seq_len:  # fires only on the prompt forward pass
            hidden[:, target_position, :] = hidden[:, target_position, :] + alpha * v_t
        if isinstance(out, tuple):
            return (hidden,) + out[1:]
        return hidden
    return hook

def steered_generate(prefix: str, v_md, s_md, alpha_factor: float,
                      injection_layer: int = INJECTION_LAYER, max_new_tokens: int = 50):
    inputs = tokenizer(prefix, return_tensors="pt").to(device)
    target_position = inputs["input_ids"].shape[1] - 1  # final token of prefix

    alpha = alpha_factor * s_md if alpha_factor != 0 else 0.0
    handle = None
    if alpha_factor != 0:
        handle = get_layer_module(injection_layer).register_forward_hook(
            make_steering_hook(v_md, alpha, target_position)
        )
    try:
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, do_sample=True, temperature=0.7, top_p=0.9,
                max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id
            )
    finally:
        if handle is not None:
            handle.remove()
    return tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# ---------------------------------------------------------------------------
# 4. Evaluation loop over a benchmark CSV (idiom or figurative-extension set)
# ---------------------------------------------------------------------------

def run_eval(benchmark_csv: str, v_md, s_md, n_samples: int = 5, out_csv: str = "eval_results.csv"):
    df = pd.read_csv(benchmark_csv)
    prefix_col = "prefix"
    rows = []
    conditions = [0.0] + ALPHA_FACTOR_GRID
    for _, row in df.iterrows():
        for alpha_factor in conditions:
            for _ in range(n_samples):
                continuation = steered_generate(row[prefix_col], v_md, s_md, alpha_factor)
                rows.append({**row.to_dict(), "alpha_factor": alpha_factor, "continuation": continuation})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return rows

# ---------------------------------------------------------------------------
# 5. LLM-as-judge prompt (adapted from paper Appendix J) — call with any
#    strong instruct model (GPT-4o, Claude, Gemma-it, etc.) via API.
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a model continuation of a
sentence ending in a figurative expression ("{expression}") interprets it
FIGURATIVELY (conventional non-compositional meaning) or LITERALLY (component
words used compositionally, outside the idiomatic/figurative frame).

Sentence prefix: "{prefix}"
Continuation: "{continuation}"

Step 1: State the conventional figurative meaning of "{expression}".
Step 2: State the literal, compositional meaning of "{expression}".
Step 3: Check coherence: is the continuation syntactically well-formed and does
it describe an imaginable situation (treating the expression as ordinary words
if needed)? If not, label INCOHERENT and stop.
Step 4: If coherent, decide whether the continuation is clearly better
supported by the figurative or literal reading. Default to FIGURATIVE if
ambiguous.

Output one label only: FIGURATIVE, LITERAL, or INCOHERENT.
"""

if __name__ == "__main__":
    # Example skeleton run (fill in real IdioLink dataframe first):
    # idiolink_df = pd.read_csv("idiolink_train_subset.csv")  # columns: sentence, label, idiom
    # v_md, s_md = build_steering_vector(idiolink_df)
    # run_eval("idiom_eval_benchmark_draft.csv", v_md, s_md, out_csv="idiom_results.csv")
    # run_eval("figurative_benchmark_draft.csv", v_md, s_md, out_csv="figurative_results.csv")
    pass
