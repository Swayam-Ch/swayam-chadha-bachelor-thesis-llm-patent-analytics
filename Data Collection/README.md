# Data Collection

This folder contains all scripts for constructing and classifying the USPTO and EPO AI patent corpora used in:

**"LLM-Assisted Patent Analytics for Mapping AI Innovation Trends"**  
Swayam Chadha: LMU Munich, 2026

---

## Pipeline Overview

### USPTO Pipeline

```
USPTO AIPD 2023          PatentsView Bulk Data        Lens.org
(1.2M patent IDs)    +   (claims + abstracts)    +   (500-patent pilot)
        │                         │                         │
        └──────────────┬──────────┘                         │
                       ▼                                     │
              merge_claims.py                                │
                       │                                     │
                       ▼                                     │
         llm_input_with_claims.csv ◄─────────────────────────┘
         (doc_id, year, title, abstract, claim1_text_truncated)
                       │
                       ▼
          qwen_vllm_classify.py
          (Qwen2.5-7B-Instruct + vLLM on A100)
                       │
                       ▼
        classifications_final.csv
        (1,209,215 patents × 14 schema dimensions)
```

### EPO Pipeline

```
Google BigQuery                EPO OPS API
patents-public-data        →   priority country lookup
(EP patents 2010–2023)         (filter_epo_priority.py)
        │                              │
        ▼                              ▼
epo_bigquery_candidates.csv   →   epo_candidates_v4.csv
(164,076 EP AI candidates)         (European-origin only)
                                        │
                                        ▼
                               qwen_classify_epo.py
                               (same model + prompt as USPTO)
                                        │
                                        ▼
                          epo_results_qwen_bigquery.csv
                          (42,801 genuine EP AI patents)
```

---

## uspto/ — USPTO Classification Pipeline

| File | Description |
|------|-------------|
| `fetch_abstracts.py` | Downloads granted patent abstracts from `g_patent_abstract.tsv` and pre-grant abstracts from `pg_published_application_abstract.tsv` via PatentsView bulk download. Output: `abstracts_all.csv` |
| `fetch_claims.py` | Downloads annual PatentsView claims files (`g_claims_YYYY.tsv`) and extracts the first independent claim for each patent in the AIPD universe. Output: `claims_claim1.csv` |
| `merge_claims.py` | Joins AIPD patent IDs, abstracts, titles, and claim text into a single LLM input file. Output: `llm_input_with_claims.csv` |
| `mockup_100.py` | Classifies a stratified 100-patent validation sample (batch 1) using GPT-4o-mini via the OpenAI API. Output: `mockup_100_results.csv` |
| `mockup_batch2.py` | Classifies the 112-patent batch 2 validation sample using GPT-4o-mini. Input: `validation/batch2_lens_complete.csv`. Output: `validation/batch2_results_v2.csv` |
| `qwen_vllm_classify.py` | Full-scale inference script. Loads Qwen2.5-7B-Instruct via vLLM, processes patents in batches of 256, checkpoints every 10,000 patents. ~8.8 patents/second on A100, ~40 hours total. Output: `classifications_final.csv` |
| `preview_results.py` | Generates summary statistics and a 20-patent readable sample from `classifications_final.csv` |
| `prompt.txt` | Full system prompt used for LLM classification (all schema definitions and decision rules) |

---

## epo/ — EPO Extension Pipeline

Requires: free Google Cloud account (BigQuery) + free EPO OPS API key from https://developers.epo.org

| File | Description |
|------|-------------|
| `bigquery_epo_patents.py` | Queries `patents-public-data.patents.publications` on Google BigQuery for EP patents (2010–2023) with English abstracts and AI-relevant CPC codes (G06N, G06V, G06F, G16H, G05B, G06Q, G10L). Output: `epo_bigquery_candidates.csv` (164,076 patents) |
| `filter_epo_priority.py` | Fetches the earliest priority country for each EP patent via the EPO OPS API and removes patents whose earliest priority was filed at the USPTO, retaining only European-origin inventions. Checkpoints every 100 patents. Output: `epo_candidates_v4.csv` (26,584 patents) |
| `qwen_classify_epo.py` | Runs Qwen2.5-7B-Instruct classification on the EPO candidate corpus using the identical prompt and schema as the USPTO run. ~3.5 hours on A100. Output: `epo_results_qwen_bigquery.csv` |

---

## validation/ — Validation Data

| File | Description |
|------|-------------|
| `validation_sample_batch2.csv` | Stratified 112-patent sample (4 genuine + 4 non-genuine per year, 2010–2023) drawn from `classifications_final.csv` for batch 2 annotation |
| `batch2_lens_complete.csv` | Batch 2 patents with titles and abstracts retrieved from Lens.org, ensuring correct text-to-ID correspondence |
| `batch2_results_v2.csv` | GPT-4o-mini classifications for all 112 batch 2 patents, including pipeline labels for Cohen's κ computation |

---

## Classification Schema (v5)

Each patent is classified across 9 LLM-extracted dimensions:

| Dimension | Values | Grounding |
|-----------|--------|-----------|
| `is_genuine_ai` | true / false | AIPD false-positive filter |
| `ai_technique` | 11 values | WIPO 2019; ACM CCS 2012 |
| `application_domain` | 21 values | WIPO 2019; CSET; ISIC |
| `data_modality` | 6 values | AIPD 2023 |
| `innovation_orientation` | fundamental / applied / both | Frascati Manual (OECD 2015); Pavitt 1984 |
| `core_ai_task` | 7 values | WIPO 2019 |
| `contribution_type` | 5 values | Pavitt 1984 |
| `training_paradigm` | 5 values | AIPD 2023 |
| `confidence` | high / medium / low | LLM self-assessment |

Plus 2 metadata dimensions: `assignee_type` (PatentsView) and `citation_counts` (Lens.org).

---

## Full-Scale Run Results

| Corpus | Patents classified | Genuine AI | Runtime | Hardware |
|--------|------------------|------------|---------|----------|
| USPTO | 1,209,215 | 207,112 (17.1%) | ~40 hours | A100-PCIE-40GB |
| EPO | 164,076 | 42,801 (26.1%) | ~3.5 hours | A100-PCIE-40GB |

**Model:** Qwen2.5-7B-Instruct | **Inference:** vLLM 0.6.3 | **Temperature:** 0 | **Max tokens:** 400

---

## Validation Summary

| Batch | n | κ | Recall | F1 |
|-------|---|---|--------|-----|
| Batch 1 | 100 | 0.820 | 1.000 | 0.901 |
| Batch 2 | 112 | 0.643 | 0.714 | 0.800 |
| **Combined** | **212** | **0.726** | **0.849** | **0.847** |

Additional dimensions validated on batch 2 (n=40 jointly genuine):
- `ai_technique` agreement: 50.0%
- `innovation_orientation` agreement: 85.0%
