"""
merge_claims.py
===============
Joins claim-1 text from fetch_claims.py output into the LLM input pipeline.

Takes:
  - aipd_years.csv             : full AIPD universe (doc_id + year)
  - lens_500_combined.csv      : titles + abstracts from Lens.org (for the 475-patent sample)
  - claims_claim1.csv          : first independent claim per patent (from fetch_claims.py)

Produces:
  - llm_input_with_claims.csv  : doc_id, year, title, abstract, claim1_text, has_claim
    → ready to feed into the LRZ Qwen classification job

Also produces a diagnostic report on coverage.

Usage:
  python merge_claims.py
"""

import os
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
AIPD_YEARS_CSV     = os.path.expanduser("~/Downloads/aipd_years.csv")
ABSTRACTS_CSV      = os.path.expanduser("~/Downloads/abstracts_all.csv")       # from fetch_abstracts.py
CLAIMS_CSV         = os.path.expanduser("~/Downloads/claims_claim1.csv")
OUTPUT_CSV         = os.path.expanduser("~/Downloads/llm_input_with_claims.csv")
SAMPLE_OUTPUT_CSV  = os.path.expanduser("~/Downloads/llm_input_sample_10.csv")


# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading files...")

aipd = pd.read_csv(AIPD_YEARS_CSV, dtype=str, low_memory=False)
aipd.columns = [c.strip() for c in aipd.columns]
aipd["doc_id"] = aipd["doc_id"].str.strip()
print(f"  AIPD universe       : {len(aipd):,} patents")

claims = pd.read_csv(CLAIMS_CSV, dtype=str)
claims["doc_id"] = claims["doc_id"].str.strip()
print(f"  Claims (claim 1)    : {len(claims):,} records")

abstracts = pd.read_csv(ABSTRACTS_CSV, dtype=str)
abstracts["doc_id"] = abstracts["doc_id"].str.strip()
print(f"  Abstracts           : {len(abstracts):,} records")


# ── MERGE ─────────────────────────────────────────────────────────────────────
print("\nMerging...")

# 1. Start with AIPD universe
df = aipd[["doc_id", "year"]].copy()

# 2. Join titles + abstracts from PatentsView bulk data
abstracts_slim = abstracts[["doc_id", "title", "abstract"]].drop_duplicates("doc_id")
df = df.merge(abstracts_slim, on="doc_id", how="left")
has_abstract = df["abstract"].notna().sum()
print(f"  Patents with abstract: {has_abstract:,} / {len(df):,} ({has_abstract/len(df)*100:.1f}%)")

# 3. Join claim 1 text
claims_slim = claims[["doc_id", "claim1_text", "claim_type"]].drop_duplicates("doc_id")
df = df.merge(claims_slim, on="doc_id", how="left")
has_claim = df["claim1_text"].notna().sum()
print(f"  Patents with claim 1 : {has_claim:,} / {len(df):,} ({has_claim/len(df)*100:.1f}%)")

# 4. Add coverage flag
df["has_abstract"] = df["abstract"].notna()
df["has_claim"]    = df["claim1_text"].notna()
df["has_both"]     = df["has_abstract"] & df["has_claim"]

# 5. Truncate claim text to ~800 chars to control LLM token usage
# (first independent claim is often very long; 800 chars captures the core scope)
MAX_CLAIM_CHARS = 800
df["claim1_text_truncated"] = df["claim1_text"].str[:MAX_CLAIM_CHARS]


# ── SAVE FULL OUTPUT ──────────────────────────────────────────────────────────
out_cols = ["doc_id", "year", "title", "abstract",
            "claim1_text_truncated", "claim_type", "has_abstract", "has_claim"]
df[out_cols].to_csv(OUTPUT_CSV, index=False)
print(f"\n  Saved full output → {OUTPUT_CSV}")


# ── DIAGNOSTIC SUMMARY ────────────────────────────────────────────────────────
print(f"\n── COVERAGE REPORT ───────────────────────────────────────────")
total = len(df)
print(f"  Total AIPD universe          : {total:>10,}")
print(f"  Has abstract (PatentsView)   : {df['has_abstract'].sum():>10,} ({df['has_abstract'].mean()*100:.1f}%)")
print(f"  Has claim 1 (PatentsView)    : {df['has_claim'].sum():>10,} ({df['has_claim'].mean()*100:.1f}%)")
print(f"  Has BOTH abstract + claim 1  : {df['has_both'].sum():>10,} ({df['has_both'].mean()*100:.1f}%)")

# Coverage by year (for patents with claims)
print(f"\n  Claim-1 coverage by year:")
year_cov = df.groupby("year")["has_claim"].agg(["sum","count"])
year_cov["pct"] = (year_cov["sum"] / year_cov["count"] * 100).round(1)
print(year_cov.rename(columns={"sum":"with_claim","count":"total"}).to_string())

# Coverage by claim type
print(f"\n  Claim type breakdown:")
print(df[df["has_claim"]]["claim_type"].value_counts().to_string())


# ── SAMPLE OUTPUT (10 patents with both abstract + claim) ─────────────────────
sample = df[df["has_both"]].head(10)[
    ["doc_id", "year", "title", "abstract", "claim1_text_truncated", "claim_type"]
].copy()
# Restore full column name for readability
sample = sample.rename(columns={"claim1_text_truncated": "claim1_text"})
sample.to_csv(SAMPLE_OUTPUT_CSV, index=False)
print(f"\n  10-patent sample → {SAMPLE_OUTPUT_CSV}")
print(f"  Review this to verify claim text quality before the LRZ run.")