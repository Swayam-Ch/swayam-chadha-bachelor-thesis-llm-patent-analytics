"""
fetch_abstracts.py
==================
Downloads PatentsView bulk files to extract title + abstract for the AIPD universe.

Sources:
  Granted titles   : g_patent.tsv.zip           -> patent_id, patent_title
  Granted abstracts: g_brf_sum_text_YYYY.tsv.zip -> patent_id, summary_text (one per year)
  Pre-grant both   : pg_published_application_abstract.tsv.zip -> pgpub_id, application_abstract
                     pg_published_application.tsv.zip           -> pgpub_id, invention_title

NOTE: PatentsView split abstracts from the main patent table in their 2022 restructure.
Granted abstracts are now in the brief summary text files (one per year, ~600MB each zipped).
Since these are large, we use claim 1 as the primary text for granted patents and only
download pre-grant abstracts + all titles.

Outputs:
  ~/Downloads/abstracts_all.csv  — doc_id, title, abstract, source
"""

import os, io, zipfile, requests, pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
AIPD_YEARS_CSV = os.path.expanduser("~/Downloads/aipd_years.csv")
OUTPUT_CSV     = os.path.expanduser("~/Downloads/abstracts_all.csv")
DOWNLOAD_DIR   = os.path.expanduser("~/Downloads/patentsview_abstracts")
YEARS          = list(range(2010, 2024))

# Known-good URLs with correct column names
GRANTED_PATENT_URL   = "https://s3.amazonaws.com/data.patentsview.org/download/g_patent.tsv.zip"
PREGRANT_ABST_URL    = "https://s3.amazonaws.com/data.patentsview.org/pregrant_publications/pg_published_application_abstract.tsv.zip"
PREGRANT_TITLE_URL   = "https://s3.amazonaws.com/data.patentsview.org/pregrant_publications/pg_published_application.tsv.zip"
# Granted abstract (brief summary) — split by year, ~600MB each zipped
GRANTED_ABST_URL     = "https://s3.amazonaws.com/data.patentsview.org/download/g_brf_sum_text_{year}.tsv.zip"

Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)

# ── LOAD AIPD UNIVERSE ────────────────────────────────────────────────────────
print("Loading AIPD universe...")
aipd = pd.read_csv(AIPD_YEARS_CSV, dtype=str, low_memory=False)
aipd["doc_id"] = aipd["doc_id"].str.strip()
granted_ids  = set(aipd[aipd["doc_id"].str.match(r"^\d{6,8}$", na=False)]["doc_id"])
pregrant_ids = set(aipd[aipd["doc_id"].str.match(r"^20\d{9}$", na=False)]["doc_id"])
reissue_ids  = set(aipd[aipd["doc_id"].str.match(r"^RE\d+", na=False)]["doc_id"])
reissue_map  = {r.replace("RE",""):r for r in reissue_ids}
granted_join = granted_ids | set(reissue_map.keys())
print(f"  Granted: {len(granted_ids):,}  Pre-grant: {len(pregrant_ids):,}  Reissue: {len(reissue_ids):,}")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def download(url, path):
    if os.path.exists(path):
        print(f"  Already have: {path}")
        return True
    print(f"  Downloading {url}")
    r = requests.get(url, stream=True, timeout=120)
    if r.status_code in (403, 404):
        print(f"  SKIP: HTTP {r.status_code}")
        return False
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    done = 0
    with open(path, "wb") as f:
        for chunk in r.iter_content(1024*1024):
            f.write(chunk); done += len(chunk)
            if total: print(f"  {done/1e6:.0f}/{total/1e6:.0f} MB ({done/total*100:.0f}%)  ", end="\r")
    print()
    return True

def stream_extract(zip_path, target_ids, id_col, title_col, abst_col, source, reissue_map=None):
    """Extract id + title + abstract from a TSV zip for target_ids."""
    rows = []
    seen = set()
    with zipfile.ZipFile(zip_path) as zf:
        tsv = next(n for n in zf.namelist() if n.endswith(".tsv"))
        print(f"  Streaming {tsv} (cols: {id_col}, {title_col}, {abst_col})")
        stream = io.TextIOWrapper(zf.open(tsv), encoding="utf-8", errors="replace")
        for chunk in pd.read_csv(stream, sep="\t", dtype=str, chunksize=100_000, on_bad_lines="skip"):
            chunk.columns = [c.strip() for c in chunk.columns]
            # Print columns on first chunk only
            if not rows and not seen:
                avail = [c for c in [id_col, title_col, abst_col] if c and c in chunk.columns]
                missing = [c for c in [id_col, title_col, abst_col] if c and c not in chunk.columns]
                if missing:
                    print(f"  WARN: missing cols {missing}. Available: {list(chunk.columns)[:8]}")
            if id_col not in chunk.columns: break
            chunk[id_col] = chunk[id_col].str.strip()
            hits = chunk[chunk[id_col].isin(target_ids) & ~chunk[id_col].isin(seen)]
            for _, row in hits.iterrows():
                raw = row[id_col]; seen.add(raw)
                rows.append({
                    "doc_id":   reissue_map.get(raw, raw) if reissue_map else raw,
                    "title":    str(row[title_col]).strip() if title_col and title_col in chunk.columns else "",
                    "abstract": str(row[abst_col]).strip()  if abst_col  and abst_col  in chunk.columns else "",
                    "source":   source
                })
    print(f"  -> {len(rows):,} records")
    return rows

# ── STEP 1: GRANTED TITLES (g_patent.tsv) ────────────────────────────────────
print("\n── Step 1: Granted titles ──────────────────────────────────────────────")
g_patent_zip = os.path.join(DOWNLOAD_DIR, "g_patent.tsv.zip")
granted_rows = []
if download(GRANTED_PATENT_URL, g_patent_zip):
    granted_rows = stream_extract(g_patent_zip, granted_join,
        id_col="patent_id", title_col="patent_title", abst_col=None,
        source="granted", reissue_map=reissue_map)

# Build granted dict: patent_id -> title
granted_titles = {r["doc_id"]: r["title"] for r in granted_rows}
print(f"  Granted titles collected: {len(granted_titles):,}")

# ── STEP 2: GRANTED ABSTRACTS (g_patent_abstract.tsv.zip) ───────────────────
print("\n── Step 2: Granted abstracts (g_patent_abstract.tsv.zip) ───────────────")
g_abst_zip = os.path.join(DOWNLOAD_DIR, "g_patent_abstract.tsv.zip")
granted_abst = {}
GRANTED_ABST_URL = "https://s3.amazonaws.com/data.patentsview.org/download/g_patent_abstract.tsv.zip"
if download(GRANTED_ABST_URL, g_abst_zip):
    rows = stream_extract(g_abst_zip, granted_join,
        id_col="patent_id", title_col=None, abst_col="patent_abstract",
        source="granted")
    granted_abst = {r["doc_id"]: r["abstract"] for r in rows}
    print(f"  Granted abstracts: {len(granted_abst):,}")

# ── STEP 3: PRE-GRANT ABSTRACTS ──────────────────────────────────────────────
print("\n── Step 3: Pre-grant abstracts ─────────────────────────────────────────")
pg_abst_zip = os.path.join(DOWNLOAD_DIR, "pg_published_application_abstract.tsv.zip")
if download(PREGRANT_ABST_URL, pg_abst_zip):
    pg_abst_rows = stream_extract(pg_abst_zip, pregrant_ids,
        id_col="pgpub_id", title_col=None, abst_col="application_abstract",
        source="pregrant")
else:
    pg_abst_rows = []
pg_abst = {r["doc_id"]: r["abstract"] for r in pg_abst_rows}
print(f"  Pre-grant abstracts: {len(pg_abst):,}")

# ── STEP 4: PRE-GRANT TITLES ─────────────────────────────────────────────────
# pg_published_application.tsv only has metadata columns, not title.
# PatentsView stores pre-grant titles in pg_publications.tsv or similar.
# Use the PatentsView API to get titles for pre-grant, OR use a workaround:
# Since we have claim 1 for all pre-grant patents, skip titles for now.
# Title will be empty for pre-grant; the LLM will use abstract + claim 1.
print("\n── Step 4: Pre-grant titles ────────────────────────────────────────────")
print("  Note: PatentsView pre-grant title table has different column structure.")
print("  Pre-grant patents will use abstract + claim 1 (title left empty).")
pg_titles = {}  # will remain empty; LLM uses abstract + claim 1 instead

# ── ASSEMBLE ──────────────────────────────────────────────────────────────────
print("\nAssembling final table...")
all_ids = list(granted_join | pregrant_ids | reissue_ids)
records = []
for doc_id in aipd["doc_id"]:
    is_granted = doc_id in granted_join or doc_id in reissue_ids
    records.append({
        "doc_id":   doc_id,
        "title":    granted_titles.get(doc_id, pg_titles.get(doc_id, "")),
        "abstract": granted_abst.get(doc_id,  pg_abst.get(doc_id, "")),
        "source":   "granted" if is_granted else "pregrant"
    })

df = pd.DataFrame(records)
df["has_title"]    = df["title"].str.strip().str.len() > 2
df["has_abstract"] = df["abstract"].str.strip().str.len() > 10

df.to_csv(OUTPUT_CSV, index=False)

print(f"\n── SUMMARY ──────────────────────────────────────────────────────────")
print(f"  Total records        : {len(df):,}")
print(f"  Has title            : {df['has_title'].sum():,} ({df['has_title'].mean()*100:.1f}%)")
print(f"  Has abstract         : {df['has_abstract'].sum():,} ({df['has_abstract'].mean()*100:.1f}%)")
print(f"  Output -> {OUTPUT_CSV}")