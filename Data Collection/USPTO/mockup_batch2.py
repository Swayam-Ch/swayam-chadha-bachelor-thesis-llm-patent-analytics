import os, json, time
import pandas as pd
import openai

OPENAI_MODEL = "gpt-4o-mini"
TEMPERATURE  = 0
INPUT_CSV    = os.path.expanduser("~/Downloads/batch2_lens_complete.csv")
OUTPUT_CSV   = os.path.expanduser("~/Downloads/batch2_results_v2.csv")

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
  - "both": ONLY when technique is specifically shaped by AND inseparable from its domain (~10-15%)
If is_genuine_ai is false, set all other fields (except false_positive_reason) to null."""

client = openai.OpenAI(api_key="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")
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} patents")

def classify(abstract, doc_id, attempt=1):
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Abstract: {str(abstract)[:800]}"},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        result = json.loads(raw)
        result["doc_id"] = doc_id
        result["status"] = "success"
        return result
    except json.JSONDecodeError as e:
        if attempt < 3:
            time.sleep(1)
            return classify(abstract, doc_id, attempt + 1)
        return {"doc_id": doc_id, "status": f"json_error: {e}"}
    except Exception as e:
        return {"doc_id": doc_id, "status": f"error: {e}"}

results = []
for i, row in df.iterrows():
    doc_id   = row["doc_id"]
    abstract = str(row.get("abstract", "")).strip()

    if not abstract or abstract == "nan":
        print(f"  [{i+1}/{len(df)}] SKIP {doc_id} — no abstract")
        results.append({"doc_id": doc_id, "status": "skipped"})
        continue

    res = classify(abstract, doc_id)
    res["pipeline_is_genuine_ai"]          = row["is_genuine_ai"]
    res["pipeline_ai_technique"]           = row.get("ai_technique")
    res["pipeline_innovation_orientation"] = row.get("innovation_orientation")
    results.append(res)
    print(f"  [{i+1}/{len(df)}] {doc_id} — genuine={res.get('is_genuine_ai')}  orient={res.get('innovation_orientation')}  conf={res.get('confidence')}")
    time.sleep(0.3)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone. Saved to {OUTPUT_CSV}")
