"""
qwen_vllm_classify.py
=====================
Full-scale patent classification using Qwen2.5-7B-Instruct + vLLM.
Expected speed: 15-30 patents/second → ~12-20 hours for 1.2M patents.

Usage:
  source ~/miniconda/bin/activate thesis
  CUDA_VISIBLE_DEVICES=0 python3 qwen_vllm_classify.py
"""

import os, json, time
import pandas as pd
from pathlib import Path
from vllm import LLM, SamplingParams

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.expanduser("~/qwen2.5-7b")
INPUT_CSV     = os.path.expanduser("~/llm_input_with_claims.csv")
RESULTS_DIR   = os.path.expanduser("~/results")
CHECKPOINT_N  = 10_000
BATCH_SIZE    = 256   # vLLM handles large batches efficiently
MAX_TOKENS    = 400

Path(RESULTS_DIR).mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are an expert AI patent analyst. Classify the following patent based on its title, abstract, and first independent claim. Respond ONLY with a valid JSON object matching the schema exactly. Do not add any explanation, preamble, or markdown fences.

SCHEMA:
{
  "is_genuine_ai": true | false,
  "false_positive_reason": "string explaining why not genuine AI, or null",
  "ai_technique": one of ["transformer_llm", "computer_vision_cnn", "classical_ml", "neural_network_general", "generative_model", "reinforcement_learning", "speech_audio", "knowledge_graph", "optimization", "ai_hardware", "not_applicable"],
  "application_domain": one of ["nlp_text", "computer_vision_imaging", "healthcare_medical", "autonomous_systems", "finance_business", "robotics", "security_privacy", "networking_infrastructure", "consumer_electronics", "industrial_manufacturing", "gaming_entertainment", "productivity_enterprise", "speech_audio", "scientific_research", "ai_methods_infrastructure", "education", "automotive", "social_media", "agriculture", "energy_management", "other"],
  "data_modality": one of ["image_video", "text_language", "audio_speech", "sensor_tabular", "multimodal", "other"],
  "innovation_orientation": one of ["fundamental", "applied", "both"],
  "core_ai_task": one of ["classification_detection", "generation", "optimization_planning", "prediction", "control", "decision_support", "other"],
  "contribution_type": one of ["algorithmic", "architectural", "system_integration", "application_implementation", "data_method"],
  "training_paradigm": one of ["supervised", "unsupervised_selfsupervised", "reinforcement_learning", "few_shot_zeroshot", "not_specified"],
  "is_llm_related": true | false,
  "confidence": one of ["high", "medium", "low"],
  "reasoning": "one sentence explaining the innovation_orientation classification only"
}

CLASSIFICATION RULES:

is_genuine_ai: true only if the core technical contribution involves AI/ML methods that learn from data. false for rule-based systems, signal processing, databases, or UI patents without a learning component.

ai_technique: Select the PRIMARY technique. Use "not_applicable" if is_genuine_ai is false.

application_domain: PRIMARY deployment domain. Use "ai_methods_infrastructure" for patents describing AI methods without a specific application.

data_modality: PRIMARY data type processed. Use "multimodal" ONLY when two or more modalities are core inputs. Use "sensor_tabular" for structured/time-series/IoT data.

innovation_orientation:
  - "fundamental": Claims protect a new AI technique/architecture applicable across domains. Could a researcher in a different domain use this directly? If yes → fundamental.
  - "applied": Claims protect a domain-specific solution where AI is the enabling tool. Would this patent exist without its domain? If no → applied.
  - "both": ONLY when technique is inseparable from its domain. Target ~10-15%. If uncertain, prefer single category.

contribution_type:
  - "algorithmic": new learning rule, loss function, or inference procedure
  - "architectural": new model architecture or layer design
  - "system_integration": combining existing AI components into novel pipeline
  - "application_implementation": applying known AI to new domain without architectural novelty
  - "data_method": novel data collection, annotation, or preprocessing

training_paradigm: PRIMARY learning paradigm for training. Use "not_specified" if only inference is described.

If is_genuine_ai is false, set all classification fields (except false_positive_reason) to null."""

SCHEMA_NULL = {
    "is_genuine_ai": None, "false_positive_reason": None,
    "ai_technique": None, "application_domain": None, "data_modality": None,
    "innovation_orientation": None, "core_ai_task": None,
    "contribution_type": None, "training_paradigm": None,
    "is_llm_related": None, "confidence": None, "reasoning": None
}

def make_prompt(row):
    title    = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    claim1   = str(row.get("claim1_text_truncated", "") or "").strip()
    parts = []
    if title:    parts.append(f"Title: {title}")
    if abstract: parts.append(f"Abstract: {abstract[:600]}")
    if claim1:   parts.append(f"First Independent Claim: {claim1[:800]}")
    user_text = "\n\n".join(parts) if parts else "No text available."
    return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"

def parse_json(text):
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"): text = text[4:]
    s, e = text.find("{"), text.rfind("}") + 1
    if s == -1 or e == 0: return None
    try: return json.loads(text[s:e])
    except: return None

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(INPUT_CSV, dtype=str, low_memory=False)
print(f"  Total: {len(df):,}")

# Resume logic
checkpoint_files = sorted(Path(RESULTS_DIR).glob("checkpoint_*.csv"))
processed_ids, prior_results = set(), []
if checkpoint_files:
    latest = checkpoint_files[-1]
    print(f"  Resuming from: {latest}")
    prior_df = pd.read_csv(latest, dtype=str)
    processed_ids = set(prior_df["doc_id"].dropna())
    prior_results = prior_df.to_dict("records")
    print(f"  Already done: {len(processed_ids):,}")

remaining = df[~df["doc_id"].isin(processed_ids)].reset_index(drop=True)
print(f"  Remaining: {len(remaining):,}")

# ── LOAD vLLM ─────────────────────────────────────────────────────────────────
print("\nLoading vLLM engine...")
llm = LLM(
    model=MODEL_PATH,
    dtype="bfloat16",
    gpu_memory_utilization=0.88,
    max_model_len=3072,
    trust_remote_code=True
)
sampling_params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
print("Engine ready.")

# ── INFERENCE ─────────────────────────────────────────────────────────────────
all_results = prior_results.copy()
total = len(remaining)
start = time.time()
print(f"\nProcessing {total:,} patents in batches of {BATCH_SIZE}...")

for batch_start in range(0, total, BATCH_SIZE):
    batch = remaining.iloc[batch_start:batch_start + BATCH_SIZE]
    prompts = [make_prompt(row) for _, row in batch.iterrows()]

    outputs = llm.generate(prompts, sampling_params)

    for i, (out, (_, row)) in enumerate(zip(outputs, batch.iterrows())):
        text = out.outputs[0].text
        parsed = parse_json(text)
        result = {"doc_id": row.get("doc_id",""), "year": row.get("year","")}
        if parsed:
            result.update(parsed); result["status"] = "success"
        else:
            result.update(SCHEMA_NULL)
            result["status"] = "parse_error"
            result["raw_response"] = text[:200]
        all_results.append(result)

    n_done = len(all_results)
    elapsed = time.time() - start
    rate = (n_done - len(prior_results)) / elapsed if elapsed > 0 else 0
    eta = (total - (n_done - len(prior_results))) / rate / 3600 if rate > 0 else 0
    print(f"  [{n_done:>8,} / {total+len(prior_results):,}]  {rate:.1f} pat/s  ETA: {eta:.1f}h", end="\r")

    # Checkpoint
    if n_done % CHECKPOINT_N < BATCH_SIZE:
        ckpt = os.path.join(RESULTS_DIR, f"checkpoint_{n_done:07d}.csv")
        pd.DataFrame(all_results).to_csv(ckpt, index=False)
        print(f"\n  ✓ Checkpoint: {ckpt}")
        for old in sorted(Path(RESULTS_DIR).glob("checkpoint_*.csv"))[:-2]:
            old.unlink()

# ── SAVE FINAL ────────────────────────────────────────────────────────────────
print("\n\nSaving final results...")
final_df = pd.DataFrame(all_results)
final_path = os.path.join(RESULTS_DIR, "classifications_final.csv")
final_df.to_csv(final_path, index=False)

success = final_df[final_df["status"] == "success"]
genuine = success[success["is_genuine_ai"].astype(str) == "True"]
print(f"\n── SUMMARY ──────────────────────────────────────────────────")
print(f"  Total     : {len(final_df):,}")
print(f"  Successful: {len(success):,} ({len(success)/len(final_df)*100:.1f}%)")
print(f"  Genuine AI: {len(genuine):,} ({len(genuine)/len(success)*100:.1f}%)")
print(f"  Time      : {(time.time()-start)/3600:.1f}h")
print(f"  Output    : {final_path}")