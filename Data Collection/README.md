# Data Collection

This folder contains all scripts and outputs for the data collection phase of the bachelor's thesis:

**"LLM-Assisted Patent Analytics for Mapping AI Innovation Trends"**  
Swayam Chadha — LMU Munich, 2026

---

## Overview

The data collection phase constructs a structured, LLM-classified dataset of 1,209,215 AI-related US patents (2010–2023) from the USPTO AI Patent Dataset (AIPD). Each patent is classified across 9 dimensions using a Qwen2.5-7B-Instruct model running on an NVIDIA A100 GPU via vLLM.

---

## Pipeline

```
USPTO AIPD 2023          PatentsView Bulk Data        Lens.org
(1.2M patent IDs)    +   (claims + abstracts)    +   (500-patent sample)
        │                         │                         │
        └──────────────┬──────────┘                         │
                       ▼                                     │
              merge_claims.py                                │
                       │                                     │
                       ▼                                     │
         llm_input_with_claims.csv ◄─────────────────────────┘
         (doc_id, year, title, abstract, claim1_text)
                       │
                       ▼
          qwen_vllm_classify.py
          (Qwen2.5-7B + vLLM on A100)
                       │
                       ▼
        classifications_final.csv
        (1,209,215 patents × 14 schema dimensions)
```

---

## Files

| File | Description |
|------|-------------|
| `fetch_claims.py` | Downloads PatentsView bulk claims TSV files (2010–2023) and extracts the first independent claim (claim_sequence=0 for granted, 1 for pre-grant) for each patent in the AIPD universe. Output: `claims_claim1.csv` |
| `fetch_abstracts.py` | Downloads `g_patent.tsv.zip` (granted titles), `g_patent_abstract.tsv.zip` (granted abstracts), and `pg_published_application_abstract.tsv.zip` (pre-grant abstracts) from PatentsView. Output: `abstracts_all.csv` |
| `merge_claims.py` | Joins `aipd_years.csv` + `abstracts_all.csv` + `claims_claim1.csv` into a single LLM input file. Output: `llm_input_with_claims.csv` |
| `mockup_100.py` | Runs the v5 annotation schema on a stratified 100-patent sample for validation before the full run. Uses OpenAI GPT-4o-mini. |
| `qwen_vllm_classify.py` | Full-scale inference script. Loads Qwen2.5-7B-Instruct via vLLM, processes 1.2M patents in batches of 256, saves checkpoints every 10,000 patents. Runs on a single A100-40GB GPU at ~8.8 patents/second (~40 hours). |
| `preview_results.py` | Generates summary statistics and a 20-patent readable sample from `classifications_final.csv`. |
| `prompt.txt` | The full system prompt used for LLM classification, with all schema definitions and decision rules. |
| `llm_input_sample_10.csv` | 10-patent sample of the LLM input file showing title + abstract + claim 1 structure. |
| `mockup_100_results.csv` | Full results from the 100-patent validation run. |
| `mockup_100_sample.csv` | 10-patent professor review sample from the mockup run. |
| `results_preview_20.csv` | 20-patent diverse sample from the full classification results. |
| `results_summary_stats.txt` | Aggregate statistics from the full 1,209,215-patent run. |

---

## Data Sources

| Source | Use | Access |
|--------|-----|--------|
| [USPTO AIPD 2023](https://www.uspto.gov/ip-policy/economic-research/research-datasets/artificial-intelligence-patent-dataset) | Patent universe definition (1,209,215 AI-related patents, 2010–2023) | Free download |
| [PatentsView Bulk Data](https://patentsview.org/download/data-download-tables) | Claim text, abstracts, titles, assignee data | Free download |
| [Lens.org](https://www.lens.org) | Abstracts + metadata for 500-patent pilot sample | Free account |

---

## Classification Schema (v5)

Each genuine AI patent is classified across 9 dimensions:

| Dimension | Values | Grounding |
|-----------|--------|-----------|
| `is_genuine_ai` | true / false | AIPD false-positive filter |
| `ai_technique` | 11 values (transformer_llm, computer_vision_cnn, classical_ml, …) | WIPO 2019 / ACM CCS |
| `application_domain` | 21 values (healthcare_medical, autonomous_systems, …) | WIPO 2019 / CSET / ISIC |
| `data_modality` | 6 values (image_video, text_language, …) | AIPD 2023; Baltrusaitis et al. 2019 |
| `innovation_orientation` | fundamental / applied / both | Frascati Manual (OECD 2015); Pavitt 1984 |
| `core_ai_task` | 7 values (classification_detection, generation, …) | WIPO 2019 Functional Applications |
| `contribution_type` | 5 values (algorithmic, architectural, …) | Pavitt 1984; Squicciarini et al. 2013 |
| `training_paradigm` | 5 values (supervised, reinforcement_learning, …) | AIPD 2023; LeCun et al. 2015 |
| `confidence` | high / medium / low | LLM self-assessment |

Two additional metadata dimensions (not LLM-extracted):
- `assignee_type` — big_tech / university / other_corp / government / individual (from PatentsView `g_assignee_disambiguated.tsv`)
- `citation_counts` — forward/backward citations (from Lens.org)

---

## Full-Scale Run Results

| Metric | Value |
|--------|-------|
| Total patents classified | 1,209,215 |
| Success rate | 100.0% |
| Genuine AI patents | 207,112 (17.1%) |
| Runtime | 40.2 hours |
| Model | Qwen2.5-7B-Instruct |
| Hardware | NVIDIA A100-PCIE-40GB |
| Inference framework | vLLM 0.6.3 |

---

## Validation

Prior to the full run, schema v5 was validated on a stratified 100-patent sample:
- JSON compliance: 100%
- Cohen's Kappa (is_genuine_ai): 0.820 ("almost perfect")
- v4 → v5 orientation agreement: 88.9%
- Cost: $0.011 for 100 patents via OpenAI GPT-4o-mini

---

## Reproducibility

```bash
# 1. Fetch claim text (PatentsView bulk download)
python fetch_claims.py

# 2. Fetch abstracts and titles
python fetch_abstracts.py

# 3. Merge into LLM input file
python merge_claims.py

# 4. Run full classification (requires GPU with vLLM)
source ~/miniconda/bin/activate thesis
CUDA_VISIBLE_DEVICES=0 python qwen_vllm_classify.py

# 5. Preview results
python preview_results.py
```

**Requirements:** Python 3.10+, vLLM 0.6.3, transformers, pandas, requests  
**GPU:** NVIDIA A100 40GB (or equivalent with ≥20GB VRAM for Qwen2.5-7B)
