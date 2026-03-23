import os, json, time, random
import pandas as pd
import openai

# ── CONFIG ────────────────────────────────────────────────────────────────────
OPENAI_MODEL       = "gpt-4o-mini"
TEMPERATURE        = 0
SAMPLE_SIZE        = 100
PROFESSOR_N        = 10
RANDOM_SEED        = 42

# Paths
CLASSIFICATIONS_CSV = os.path.expanduser("~/Downloads/llm_500_final_corrected.csv")  # v4 classifications + patent_id
BRIDGE_CSV          = os.path.expanduser("~/Downloads/lens_years_bridge.csv")         # Lens ID <-> patent_id <-> year
LENS_COMBINED_CSV   = os.path.expanduser("~/Downloads/lens_500_combined.csv")         # Title + Abstract
OUTPUT_FULL         = os.path.expanduser("~/Downloads/mockup_100_results.csv")
OUTPUT_SAMPLE       = os.path.expanduser("~/Downloads/mockup_100_professor_sample.csv")

LENS_API_KEY       = os.environ.get("LENS_API_KEY", "")
LENS_API_URL       = "https://api.lens.org/patent/search"

# ── PROMPT ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI patent analyst. Classify the following patent based solely on its title and abstract. Respond ONLY with a valid JSON object matching the schema exactly. Do not add any explanation, preamble, or markdown fences.

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

is_genuine_ai: true only if the core technical contribution involves AI/ML methods. false for patents that merely use AI as a peripheral component, or involve signal processing, databases, or UI without a learning component.

ai_technique: Select the PRIMARY technique that is the core of the invention. If multiple apply, choose the most specific. Use "neural_network_general" only when no specific architecture is identifiable.

application_domain: Select the PRIMARY domain where the invention is deployed. Use "ai_methods_infrastructure" for patents describing AI methods without a specific application (model training, inference optimization, etc.).

data_modality: Select the PRIMARY data type the AI processes. Use "multimodal" ONLY when two or more modalities are explicitly mentioned as core inputs to the same model. Use "sensor_tabular" for structured data, time-series, IoT, and numerical inputs.

innovation_orientation:
  - "fundamental": Claims primarily protect a new AI technique, architecture, training method, or algorithm that could apply across domains. The domain mentioned is an illustration, not a constraint. Ask: could a researcher in a completely different domain use this directly? If yes → fundamental.
  - "applied": Claims primarily protect a solution to a domain-specific problem where AI is the enabling tool. The invention would not exist without its domain context. Ask: would this patent exist without its domain? If no → applied.
  - "both": ONLY when the technique is specifically shaped by AND inseparable from its domain. Target ~10-15% of genuine AI patents. If uncertain, prefer a single category.

core_ai_task: What the AI is DOING, not the domain. Use "decision_support" for systems that recommend or augment human choices without autonomous action.

contribution_type:
  - "algorithmic": new learning rule, loss function, objective, or inference procedure
  - "architectural": new model architecture, layer design, or attention mechanism
  - "system_integration": combining existing AI components into a novel pipeline
  - "application_implementation": applying known AI to a new domain task without architectural/algorithmic novelty
  - "data_method": novel data collection, annotation, augmentation, or preprocessing

training_paradigm: The PRIMARY learning paradigm used to TRAIN the model. If the patent describes inference only with no training described, use "not_specified". Note: if RL is the application technique (e.g. a game-playing agent), capture it in ai_technique — use "reinforcement_learning" here only when RL is how the model is trained.

If is_genuine_ai is false, set all other fields (except false_positive_reason) to null."""

USER_TEMPLATE = "Title: {title}\n\nAbstract: {abstract}"


# ── STEP 1: LOAD & MERGE EXISTING DATA ───────────────────────────────────────
print("Loading existing data...")

# Load v4 classifications (patent_id = Lens ID format)
merged = pd.read_csv(CLASSIFICATIONS_CSV)
print(f"  Classifications: {len(merged)} rows from {CLASSIFICATIONS_CSV}")

# patent_id in llm_500_final_corrected.csv is the Lens ID — rename directly
merged = merged.rename(columns={"patent_id": "Lens ID"})

# Join year from bridge (Lens ID -> year)
bridge = pd.read_csv(BRIDGE_CSV)[["Lens ID", "year"]].drop_duplicates("Lens ID")
merged = merged.merge(bridge, on="Lens ID", how="left")
print(f"  After year join: {merged['year'].notna().sum()}/{len(merged)} have year")
print(f"  Columns: {list(merged.columns)}")


# ── STEP 2: GET TITLES + ABSTRACTS ───────────────────────────────────────────
def load_abstracts_from_lens_csv(path):
    """Load from local lens_500_combined.csv if available."""
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} records from {path}")
    print(f"  Columns: {list(df.columns)}")
    # Lens.org CSV columns: 'Lens ID', 'Title', 'Abstract', 'Publication Year', etc.
    title_col    = next((c for c in df.columns if 'title' in c.lower()), None)
    abstract_col = next((c for c in df.columns if 'abstract' in c.lower()), None)
    id_col       = next((c for c in df.columns if 'lens' in c.lower() and 'id' in c.lower()), None)
    if not all([title_col, abstract_col, id_col]):
        raise ValueError(f"Could not find required columns. Found: {list(df.columns)}")
    return df[[id_col, title_col, abstract_col]].rename(columns={
        id_col: 'Lens ID', title_col: 'title', abstract_col: 'abstract'
    })


def fetch_abstracts_from_api(lens_ids, api_key, batch_size=50):
    """Fetch titles + abstracts from Lens.org API for given Lens IDs."""
    import requests
    results = []
    for i in range(0, len(lens_ids), batch_size):
        batch = lens_ids[i:i+batch_size]
        payload = {
            "query": {"terms": {"lens_id": batch}},
            "include": ["lens_id", "title", "abstract"],
            "size": batch_size
        }
        resp = requests.post(
            LENS_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        for hit in data.get("data", []):
            results.append({
                "Lens ID": hit.get("lens_id", ""),
                "title":   hit.get("title", ""),
                "abstract": hit.get("abstract", ""),
            })
        print(f"  Fetched batch {i//batch_size + 1}: {len(results)} total so far")
        time.sleep(0.5)  # rate limit
    return pd.DataFrame(results)


# Normalize column names: find whichever col contains 'lens'+'id' and rename to 'Lens ID'
def normalize_lens_col(frame, label):
    for col in frame.columns:
        if "lens" in col.lower() and "id" in col.lower():
            if col != "Lens ID":
                print(f"  [{label}] Renaming column '{col}' -> 'Lens ID'")
            return frame.rename(columns={col: "Lens ID"})
    raise KeyError(f"[{label}] No Lens ID column found. Columns: {list(frame.columns)}")

print("\nLoading titles + abstracts...")
if os.path.exists(LENS_COMBINED_CSV):
    print(f"  Found local file: {LENS_COMBINED_CSV}")
    text_df = load_abstracts_from_lens_csv(LENS_COMBINED_CSV)
else:
    print(f"  lens_500_combined.csv not found. Fetching from Lens.org API...")
    if not LENS_API_KEY:
        raise EnvironmentError(
            "LENS_API_KEY not set and lens_500_combined.csv not found.\n"
            "Either set export LENS_API_KEY='...' or place lens_500_combined.csv at:\n"
            f"  {LENS_COMBINED_CSV}"
        )
    all_lens_ids = merged["Lens ID"].dropna().tolist()
    text_df = fetch_abstracts_from_api(all_lens_ids, LENS_API_KEY)

text_df = normalize_lens_col(text_df, "text_df")
print(f"  merged  Lens ID sample: {merged['Lens ID'].head(2).tolist()}")
print(f"  text_df Lens ID sample: {text_df['Lens ID'].head(2).tolist()}")


# Merge text into merged dataframe
df = merged.merge(text_df, on="Lens ID", how="left")
missing_text = df["abstract"].isna().sum()
print(f"  Merged: {len(df)} rows, {missing_text} missing abstracts")
if missing_text > 0:
    print(f"  Warning: {missing_text} patents have no abstract — they will be skipped in sampling")
df = df[df["abstract"].notna() & (df["abstract"].str.strip() != "")]


# ── STEP 3: STRATIFIED SAMPLE ─────────────────────────────────────────────────
print(f"\nBuilding stratified sample (n={SAMPLE_SIZE})...")
random.seed(RANDOM_SEED)

# Stratify by year — proportional to distribution in full dataset
year_counts = df["year"].value_counts().sort_index()
total = len(df)
year_alloc = (year_counts / total * SAMPLE_SIZE).round().astype(int)

# Fix rounding to exactly hit SAMPLE_SIZE
diff = SAMPLE_SIZE - year_alloc.sum()
if diff != 0:
    # add/subtract from the largest year(s)
    for yr in year_alloc.nlargest(abs(diff)).index:
        year_alloc[yr] += (1 if diff > 0 else -1)
        diff += (-1 if diff > 0 else 1)
        if diff == 0:
            break

print(f"  Year allocation:")
for yr, n in year_alloc.items():
    print(f"    {yr}: {n}")

sample_rows = []
for yr, n in year_alloc.items():
    yr_pool = df[df["year"] == yr]
    n = min(n, len(yr_pool))
    if n > 0:
        sample_rows.append(yr_pool.sample(n, random_state=RANDOM_SEED))

sample = pd.concat(sample_rows).reset_index(drop=True)
print(f"  Final sample size: {len(sample)}")


# ── STEP 4: RUN LLM ──────────────────────────────────────────────────────────
client = openai.OpenAI()

SCHEMA_FIELDS = [
    "is_genuine_ai", "false_positive_reason",
    "ai_technique", "application_domain", "data_modality",
    "innovation_orientation", "core_ai_task", "contribution_type",
    "training_paradigm", "is_llm_related", "confidence", "reasoning"
]

def classify_patent(title: str, abstract: str, patent_id: str, attempt: int = 1) -> dict:
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(
                    title=title, abstract=abstract
                )},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if model added them anyway
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        result["patent_id"] = patent_id
        result["raw_response"] = resp.choices[0].message.content
        result["status"] = "success"
        return result
    except json.JSONDecodeError as e:
        if attempt < 3:
            print(f"    JSON parse error on {patent_id}, retrying ({attempt+1}/3)...")
            time.sleep(1)
            return classify_patent(title, abstract, patent_id, attempt + 1)
        return {"patent_id": patent_id, "status": f"json_error: {e}", "raw_response": ""}
    except Exception as e:
        return {"patent_id": patent_id, "status": f"error: {e}", "raw_response": ""}


print(f"\nRunning LLM classification on {len(sample)} patents...")
print(f"  Model: {OPENAI_MODEL}, temperature: {TEMPERATURE}")
print(f"  Estimated cost: ~${len(sample) * 0.00011:.3f}")
print()

results = []
for i, row in sample.iterrows():
    pid   = row.get("patent_id", row.get("Lens ID", f"patent_{i}"))
    title = str(row.get("title", "")).strip()
    abstr = str(row.get("abstract", "")).strip()

    if not title and not abstr:
        print(f"  [{i+1}/{len(sample)}] SKIP {pid} — no text")
        results.append({"patent_id": pid, "status": "skipped_no_text"})
        continue

    print(f"  [{i+1}/{len(sample)}] {pid[:40]}...", end=" ", flush=True)
    res = classify_patent(title, abstr, pid)
    res["year"]     = row.get("year")
    res["title"]    = title
    res["abstract"] = abstr[:300] + "..." if len(abstr) > 300 else abstr

    # Carry forward old v4 classifications for comparison
    res["v4_is_genuine_ai"]        = row.get("is_genuine_ai")
    res["v4_ai_technique"]         = row.get("ai_technique")
    res["v4_innovation_orientation"] = row.get("innovation_orientation")

    results.append(res)
    print(f"✓  genuine={res.get('is_genuine_ai')}  orientation={res.get('innovation_orientation')}  confidence={res.get('confidence')}")
    time.sleep(0.3)  # stay well within rate limits


# ── STEP 5: SAVE FULL RESULTS ─────────────────────────────────────────────────
print(f"\nSaving results...")
results_df = pd.DataFrame(results)

# Reorder columns sensibly
col_order = [
    "patent_id", "year", "title", "abstract", "status",
    "is_genuine_ai", "false_positive_reason",
    "ai_technique", "application_domain", "data_modality",
    "innovation_orientation", "core_ai_task", "contribution_type",
    "training_paradigm", "is_llm_related", "confidence", "reasoning",
    "v4_is_genuine_ai", "v4_ai_technique", "v4_innovation_orientation",
    "raw_response"
]
col_order = [c for c in col_order if c in results_df.columns]
results_df = results_df[col_order]
results_df.to_csv(OUTPUT_FULL, index=False)
print(f"  Saved {len(results_df)} rows → {OUTPUT_FULL}")


# ── STEP 6: QUICK SUMMARY STATS ──────────────────────────────────────────────
success = results_df[results_df["status"] == "success"]
genuine = success[success["is_genuine_ai"] == True]

print(f"\n── SUMMARY ──────────────────────────────────────────────────────")
print(f"  Processed: {len(success)} / {len(results_df)} successful")
print(f"  Genuine AI: {len(genuine)} ({len(genuine)/len(success)*100:.1f}%)")
if len(genuine) > 0:
    print(f"\n  Innovation Orientation:")
    print(genuine["innovation_orientation"].value_counts().to_string())
    print(f"\n  Core AI Task:")
    print(genuine["core_ai_task"].value_counts().to_string())
    print(f"\n  Contribution Type:")
    print(genuine["contribution_type"].value_counts().to_string())
    print(f"\n  Training Paradigm:")
    print(genuine["training_paradigm"].value_counts().to_string())
    print(f"\n  v4 vs v5 orientation agreement:")
    both = genuine[genuine["v4_innovation_orientation"].notna()]
    if len(both) > 0:
        agree = (both["innovation_orientation"] == both["v4_innovation_orientation"]).sum()
        print(f"    {agree}/{len(both)} agree ({agree/len(both)*100:.1f}%)")


# ── STEP 7: PROFESSOR SAMPLE (10 patents) ─────────────────────────────────────
# All genuine AI patents from this run + enough non-genuine to reach 10
# Shows professor: full range of orientations, techniques, and a false positive
print(f"\nBuilding professor review sample (n={PROFESSOR_N})...")

sample_parts = []

# Take ALL genuine AI patents (likely <10 in a proportional sample)
if len(genuine) > 0:
    sample_parts.append(genuine)

# Fill remaining slots with diverse non-genuine patents
n_nongenuine_needed = max(1, PROFESSOR_N - len(genuine))
non_genuine = success[success["is_genuine_ai"] == False]
if len(non_genuine) > 0:
    # Pick non-genuine with varied false_positive_reason if possible
    non_genuine_diverse = non_genuine.drop_duplicates(subset=["false_positive_reason"])
    n_to_take = min(n_nongenuine_needed, len(non_genuine_diverse))
    sample_parts.append(non_genuine_diverse.sample(n_to_take, random_state=RANDOM_SEED))

professor_sample = pd.concat(sample_parts).head(PROFESSOR_N).reset_index(drop=True)
print(f"  Genuine AI in sample: {professor_sample['is_genuine_ai'].sum()}")
print(f"  Non-genuine in sample: {(~professor_sample['is_genuine_ai']).sum()}")

# For the professor sample: include full abstract (not truncated) and raw_response
prof_cols = [
    "patent_id", "Lens ID", "year", "title", "abstract",
    "is_genuine_ai", "false_positive_reason",
    "ai_technique", "application_domain", "data_modality",
    "innovation_orientation", "core_ai_task", "contribution_type",
    "training_paradigm", "is_llm_related", "confidence", "reasoning",
    "raw_response"
]
prof_cols = [c for c in prof_cols if c in professor_sample.columns]
professor_sample[prof_cols].to_csv(OUTPUT_SAMPLE, index=False)
print(f"  Saved {len(professor_sample)} patents → {OUTPUT_SAMPLE}")

print(f"\n── DONE ─────────────────────────────────────────────────────────")
print(f"  Full results:       {OUTPUT_FULL}")
print(f"  Professor sample:   {OUTPUT_SAMPLE}")
print(f"\n  Next steps:")
print(f"  1. Review mockup_100_professor_sample.csv — check each classification")
print(f"  2. Note any disagreements with 'reasoning' field")
print(f"  3. Share with professor alongside the annotated prompt document")