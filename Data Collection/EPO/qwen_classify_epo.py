"""
qwen_classify_epo.py
Classifies EPO AI-candidate patents using Qwen2.5-7B-Instruct via vLLM.
Same model, prompt, schema, and inference parameters as the USPTO run.

Input:  epo_candidates_clean.csv  (ep_id, year, cpc, title, abstract)
Output: epo_results_qwen.csv

Usage:
    source ~/miniconda/bin/activate thesis
    CUDA_VISIBLE_DEVICES=0 python qwen_classify_epo.py

Requirements: vLLM 0.6.3, transformers, pandas
"""

import os, json, time, csv
import pandas as pd
from vllm import LLM, SamplingParams

MODEL_PATH    = "Qwen/Qwen2.5-7B-Instruct"
INPUT_CSV     = "epo_candidates_clean.csv"
OUTPUT_CSV    = "epo_results_qwen.csv"
CHECKPOINT_N  = 1000
BATCH_SIZE    = 256

SYSTEM_PROMPT = """You are an expert AI patent analyst. Classify the following patent based solely on its abstract. Respond ONLY with a valid JSON object matching the schema exactly. Do not add any explanation, preamble, or markdown fences.

SCHEMA:
{
  "is_genuine_ai": true | false,
  "false_positive_reason": "brief reason if not genuine AI, or null",
  "ai_technique": one of ["transformer_llm", "computer_vision_cnn", "classical_ml", "neural_network_general", "generative_model", "reinforcement_learning", "speech_audio", "knowledge_graph", "optimization", "ai_hardware", "not_applicable"],
  "application_domain": one of ["nlp_text", "computer_vision_imaging", "healthcare_medical", "autonomous_systems", "finance_business", "robotics", "security_privacy", "networking_infrastructure", "consumer_electronics", "industrial_manufacturing", "gaming_entertainment", "productivity_enterprise", "speech_audio", "scientific_research", "ai_methods_infrastructure", "education", "automotive", "social_media", "agriculture", "energy_management", "other"],
  "innovation_orientation": one of ["fundamental", "applied", "both"],
  "contribution_type": one of ["algorithmic", "architectural", "system_integration", "application_implementation", "data_method"],
  "is_llm_related": true | false,
  "confidence": one of ["high", "medium", "low"],
  "reasoning": "one sentence explaining the innovation_orientation classification only"
}

CLASSIFICATION RULES:
is_genuine_ai: true only if the core technical contribution involves AI/ML methods. false for patents that merely use AI as a peripheral component, or involve signal processing, databases, or UI without a learning component.
ai_technique: Select the PRIMARY technique. Use "neural_network_general" only when no specific architecture is identifiable.
innovation_orientation:
  - "fundamental": claims protect a new AI technique/architecture/algorithm usable across domains
  - "applied": claims protect a domain-specific solution where AI is the enabling tool
  - "both": ONLY when the technique is specifically shaped by AND inseparable from its domain (~10-15%)
If is_genuine_ai is false, set all other fields (except false_positive_reason) to null."""


def make_prompt(title, abstract):
    title    = str(title or "").strip()
    abstract = str(abstract or "").strip()[:800]
    content  = f"Title: {title}\nAbstract: {abstract}" if title else f"Abstract: {abstract}"
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_result(text, ep_id):
    try:
        raw = text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        result["ep_id"]  = ep_id
        result["status"] = "success"
        return result
    except Exception as e:
        return {"ep_id": ep_id, "status": f"parse_error: {e}", "raw": text[:200]}


def save(results, fieldnames):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} EP patents")

    # Resume from checkpoint
    results       = []
    processed_ids = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                results.append(row)
                processed_ids.add(row["ep_id"])
        print(f"Resuming: {len(results)} already classified")

    # Filter to unprocessed
    todo = df[~df["ep_id"].astype(str).isin(processed_ids)].copy()
    print(f"Remaining: {len(todo)} patents to classify")

    if len(todo) == 0:
        print("All done.")
        return

    # Load model
    print("Loading Qwen2.5-7B-Instruct via vLLM...")
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=2048,
    )
    sampling = SamplingParams(
        temperature=0,
        max_tokens=400,
        stop=["<|im_end|>"],
    )
    print("Model loaded.")

    fieldnames = [
        "ep_id", "year", "cpc",
        "is_genuine_ai", "false_positive_reason", "ai_technique",
        "application_domain", "innovation_orientation", "contribution_type",
        "is_llm_related", "confidence", "reasoning", "status"
    ]

    # Process in batches
    rows   = todo.to_dict("records")
    total  = len(rows)
    t_start = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        batch    = rows[batch_start: batch_start + BATCH_SIZE]
        prompts  = [make_prompt(r.get("title"), r.get("abstract")) for r in batch]
        outputs  = llm.generate(prompts, sampling)

        for row, out in zip(batch, outputs):
            ep_id  = str(row["ep_id"])
            text   = out.outputs[0].text
            result = parse_result(text, ep_id)
            result["year"] = row.get("year", "")
            result["cpc"]  = row.get("cpc", "")
            results.append(result)

        n_done   = batch_start + len(batch)
        elapsed  = time.time() - t_start
        rate     = n_done / elapsed
        remaining = (total - n_done) / rate if rate > 0 else 0
        print(f"  [{n_done}/{total}] {rate:.1f} patents/s — "
              f"ETA {remaining/60:.1f} min")

        if n_done % CHECKPOINT_N == 0 or n_done == total:
            save(results, fieldnames)
            print(f"  Checkpoint saved: {len(results)} total")

    save(results, fieldnames)
    success = [r for r in results if r.get("status") == "success"]
    genuine = [r for r in success if str(r.get("is_genuine_ai")).lower() == "true"]
    print(f"\nDone.")
    print(f"Classified: {len(success)}/{len(results)}")
    print(f"Genuine AI: {len(genuine)} ({len(genuine)/len(success)*100:.1f}%)")
    print(f"Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
