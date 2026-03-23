"""
preview_results.py
==================
Generates a readable 20-patent sample and summary stats from classifications_final.csv.
Run locally after downloading from the server.

Usage:
  python3 preview_results.py
"""

import pandas as pd
import os

INPUT_CSV     = os.path.expanduser("~/Downloads/classifications_final.csv")
SAMPLE_OUTPUT = os.path.expanduser("~/Downloads/results_preview_20.csv")
STATS_OUTPUT  = os.path.expanduser("~/Downloads/results_summary_stats.txt")

print("Loading results...")
df = pd.read_csv(INPUT_CSV, dtype=str, low_memory=False)
print(f"  Total patents: {len(df):,}")

success  = df[df["status"] == "success"]
genuine  = success[success["is_genuine_ai"] == "True"]
nongenuine = success[success["is_genuine_ai"] == "False"]

# ── SUMMARY STATS ─────────────────────────────────────────────────────────────
lines = []
lines.append("═" * 60)
lines.append("  FULL PIPELINE RESULTS — SUMMARY STATISTICS")
lines.append("═" * 60)
lines.append(f"\n  Total patents classified : {len(df):>10,}")
lines.append(f"  Successful               : {len(success):>10,} ({len(success)/len(df)*100:.1f}%)")
lines.append(f"  Genuine AI               : {len(genuine):>10,} ({len(genuine)/len(success)*100:.1f}%)")
lines.append(f"  Non-genuine (filtered)   : {len(nongenuine):>10,} ({len(nongenuine)/len(success)*100:.1f}%)")

lines.append("\n── AI Technique (top 10) ─────────────────────────────────")
lines.append(genuine["ai_technique"].value_counts().head(10).to_string())

lines.append("\n── Application Domain (top 10) ───────────────────────────")
lines.append(genuine["application_domain"].value_counts().head(10).to_string())

lines.append("\n── Innovation Orientation ────────────────────────────────")
lines.append(genuine["innovation_orientation"].value_counts().to_string())

lines.append("\n── Core AI Task ──────────────────────────────────────────")
lines.append(genuine["core_ai_task"].value_counts().to_string())

lines.append("\n── Contribution Type ─────────────────────────────────────")
lines.append(genuine["contribution_type"].value_counts().to_string())

lines.append("\n── Training Paradigm ─────────────────────────────────────")
lines.append(genuine["training_paradigm"].value_counts().to_string())

lines.append("\n── Confidence ────────────────────────────────────────────")
lines.append(genuine["confidence"].value_counts().to_string())

lines.append("\n── Genuine AI by Year ────────────────────────────────────")
year_counts = genuine.groupby("year").size().sort_index()
lines.append(year_counts.to_string())

stats_text = "\n".join(lines)
print(stats_text)

with open(STATS_OUTPUT, "w") as f:
    f.write(stats_text)
print(f"\n  Stats saved → {STATS_OUTPUT}")

# ── 20-PATENT SAMPLE ──────────────────────────────────────────────────────────
# Pick diverse sample: mix of orientations, techniques, years, confidence levels
sample_parts = []

# 5 fundamental
fund = genuine[genuine["innovation_orientation"] == "fundamental"]
if len(fund) >= 5:
    sample_parts.append(fund.sample(5, random_state=42))

# 10 applied — spread across techniques
applied = genuine[genuine["innovation_orientation"] == "applied"]
if len(applied) >= 10:
    applied_diverse = applied.drop_duplicates(subset=["ai_technique"])
    n = min(10, len(applied_diverse))
    sample_parts.append(applied_diverse.sample(n, random_state=42))

# 2 both
both = genuine[genuine["innovation_orientation"] == "both"]
if len(both) >= 2:
    sample_parts.append(both.sample(2, random_state=42))

# 3 non-genuine (to show false positive handling)
sample_parts.append(nongenuine.sample(min(3, len(nongenuine)), random_state=42))

sample = pd.concat(sample_parts).head(20).reset_index(drop=True)

# Select readable columns
cols = ["doc_id", "year", "is_genuine_ai", "false_positive_reason",
        "ai_technique", "application_domain", "data_modality",
        "innovation_orientation", "core_ai_task", "contribution_type",
        "training_paradigm", "is_llm_related", "confidence", "reasoning", "status"]
cols = [c for c in cols if c in sample.columns]
sample[cols].to_csv(SAMPLE_OUTPUT, index=False)
print(f"  20-patent sample saved → {SAMPLE_OUTPUT}")
print(f"\nDone. Share results_preview_20.csv with your professor.")