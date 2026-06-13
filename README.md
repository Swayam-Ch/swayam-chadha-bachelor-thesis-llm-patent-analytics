# swayam-chadha-bachelor-thesis-llm-patent-analytics

**LLM-Assisted Patent Analytics for Mapping AI Innovation Trends**  
Swayam Chadha: BSc Computer Science, LMU Munich, 2026  
Supervisor: Prof. Dr. Stefan Feuerriegel, Chair of AI in Management

---

## Overview

This repository contains all code, prompts, validation data, and figures for my bachelor's thesis. The thesis applies a large language model pipeline to classify 1,209,215 AI-candidate USPTO patents (2010–2023) across nine dimensions, and extends the same pipeline to 164,076 EP AI-candidate patents for a cross-jurisdictional comparison.

**Key results:**
- 207,112 genuine AI patents identified (17.1% of AIPD universe), κ = 0.726 on a 212-patent gold standard
- 84.5% of genuine AI patents are applied rather than fundamental, declining from 87.1% (2010) to 80.9% (2023)
- Neural network patents dominate (56.0%); classical ML declined from 16.3% to 5.2%
- EPO applied share (87.3%) closely mirrors USPTO, converging at ~81% by 2023
- Transformer patents underrepresented at EPO (5.4% vs 10.5%), classical ML near-absent (0.4% vs 8.9%)

---

## Repository Structure

```
swayam-chadha-bachelor-thesis-llm-patent-analytics/
│
├── data_collection/
│   ├── uspto/                  # USPTO classification pipeline
│   │   ├── fetch_abstracts.py
│   │   ├── fetch_claims.py
│   │   ├── merge_claims.py
│   │   ├── mockup_100.py
│   │   ├── mockup_batch2.py
│   │   ├── qwen_vllm_classify.py
│   │   ├── preview_results.py
│   │   └── prompt.txt
│   ├── epo/                    # EPO extension pipeline
│   │   ├── bigquery_epo_patents.py
│   │   ├── filter_epo_priority.py
│   │   └── qwen_classify_epo.py
│   └── validation/             # Validation data files
│       ├── validation_sample_batch2.csv
│       ├── batch2_lens_complete.csv
│       └── batch2_results_v2.csv
│
├── figure_generation/
│   ├── generate_figures.py     # Reproduces all 7 figures
│   └── figures/                # Pre-generated PDFs
│       ├── fig1_growth.pdf
│       ├── fig2_technique.pdf
│       ├── fig3_domains.pdf
│       ├── fig4_orientation.pdf
│       ├── fig5_heatmap.pdf
│       ├── fig6_epo_comparison.pdf
│       └── fig7_emerging_tech.pdf
│
├── thesis/
│   ├── thesis_main.tex
│   └── references.bib
│
└── README.md
```

---

## Reproducing the Results

### Requirements

```bash
pip install vllm==0.6.3 transformers pandas requests \
            google-cloud-bigquery db-dtypes pyarrow \
            matplotlib scipy numpy
```

**GPU:** NVIDIA A100 40 GB or equivalent with ≥20 GB VRAM for Qwen2.5-7B-Instruct.  
**Python:** 3.10+

### USPTO Pipeline

```bash
cd data_collection/uspto

# 1. Fetch abstracts and titles from PatentsView
python fetch_abstracts.py

# 2. Fetch first independent claims from PatentsView
python fetch_claims.py

# 3. Merge into LLM input file
python merge_claims.py

# 4. Run full classification (~40 hours on A100)
source ~/miniconda/bin/activate thesis
CUDA_VISIBLE_DEVICES=0 python qwen_vllm_classify.py

# 5. Preview results
python preview_results.py
```

### EPO Pipeline

```bash
cd data_collection/epo

# 1. Fetch EP AI-candidate patents via Google BigQuery (free account required)
# Edit bigquery_epo_patents.py to set your GCP project ID, then:
gcloud auth application-default login
python bigquery_epo_patents.py

# 2. Filter to European-origin patents (EPO OPS API key required)
# Register free at https://developers.epo.org, then:
export EPO_CLIENT_ID="your_key"
export EPO_CLIENT_SECRET="your_secret"
python filter_epo_priority.py

# 3. Run classification on GPU (~3.5 hours on A100)
CUDA_VISIBLE_DEVICES=0 python qwen_classify_epo.py
```

### Figures

```bash
cd figure_generation

# Place classifications_final.csv and epo_results_qwen_bigquery.csv here
python generate_figures.py
# Outputs: figures/fig1_growth.pdf ... figures/fig7_emerging_tech.pdf
```

---

## Data Sources

| Source | Use | Access |
|--------|-----|--------|
| [USPTO AIPD 2023](https://www.uspto.gov/ip-policy/economic-research/research-datasets/artificial-intelligence-patent-dataset) | 1,209,215 AI-candidate patent IDs | Free download |
| [PatentsView Bulk Data](https://patentsview.org/download/data-download-tables) | Abstracts, titles, claim text | Free download |
| [Google BigQuery patents-public-data](https://console.cloud.google.com/bigquery) | EP patent abstracts and CPC codes | Free (1 TB/month) |
| [EPO Open Patent Services](https://developers.epo.org) | Priority country lookup for EP patents | Free registration |
| [Lens.org](https://www.lens.org) | Abstracts for validation sample | Free account |

---

## Model

| Component | Value |
|-----------|-------|
| Model | Qwen2.5-7B-Instruct |
| Inference framework | vLLM 0.6.3 |
| Hardware | NVIDIA A100-PCIE-40GB |
| Temperature | 0 (greedy decoding) |
| Max output tokens | 400 |
| Passes per patent | 1 |

---

## Citation

```bibtex
@thesis{chadha2026,
  author  = {Chadha, Swayam},
  title   = {LLM-Assisted Patent Analytics for Mapping AI Innovation Trends},
  school  = {Ludwig-Maximilians-Universit{\"a}t M{\"u}nchen},
  year    = {2026},
  type    = {Bachelor's Thesis},
}
```
