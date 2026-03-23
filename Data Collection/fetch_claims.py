import os
import io
import zipfile
import requests
import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
AIPD_YEARS_CSV   = os.path.expanduser("~/Downloads/aipd_years.csv")
OUTPUT_CSV       = os.path.expanduser("~/Downloads/claims_claim1.csv")
DOWNLOAD_DIR     = os.path.expanduser("~/Downloads/patentsview_claims")
YEARS            = list(range(2010, 2024))          # 2010–2023 inclusive
KEEP_ZIPS        = False                             # set True to keep zip files after extraction

# PatentsView URL patterns — correct base is data.patentsview.org (not patentsview.org)
# Primary S3 bucket (confirmed working as of early 2026)
GRANTED_URL   = "https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_{year}.tsv.zip"
PREGRANT_URL  = "https://s3.amazonaws.com/data.patentsview.org/pregrant_publications/pg_claims_{year}.tsv.zip"

# NOTE: PatentsView is migrating to USPTO Open Data Portal (data.uspto.gov) on March 20, 2026.
# If the above URLs 404, the files are also available via Academic Torrents (full mirror):
# https://academictorrents.com/details/2c6eb904b11a8e188c59e5e5ffdd06562950d84b
# Or manually download from: https://patentsview.org/download/claims
# and place as ~/Downloads/patentsview_claims/g_claims_YYYY.tsv.zip

# ── STEP 1: LOAD AIPD UNIVERSE & SPLIT BY ID TYPE ────────────────────────────
print("Loading AIPD universe...")
aipd = pd.read_csv(AIPD_YEARS_CSV, dtype=str, low_memory=False)
aipd.columns = [c.strip() for c in aipd.columns]
id_col = aipd.columns[0]  # 'doc_id'
aipd[id_col] = aipd[id_col].str.strip()

# Categorise doc_ids
granted_ids  = set(aipd[aipd[id_col].str.match(r'^\d{6,8}$', na=False)][id_col])
pregrant_ids = set(aipd[aipd[id_col].str.match(r'^20\d{9}$', na=False)][id_col])
reissue_ids  = set(aipd[aipd[id_col].str.match(r'^RE\d+$', na=False)][id_col])

# Reissue patents are in the granted claims table; strip RE prefix for joining
reissue_numeric = {rid.replace("RE", ""): rid for rid in reissue_ids}
granted_ids_for_join = granted_ids | set(reissue_numeric.keys())

print(f"  Granted patents    : {len(granted_ids):>10,}")
print(f"  Pre-grant publs.   : {len(pregrant_ids):>10,}")
print(f"  Reissue patents    : {len(reissue_ids):>10,}")
print(f"  Total              : {len(aipd):>10,}")

# ── SETUP ─────────────────────────────────────────────────────────────────────
Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
results = []   # list of dicts: {patent_id, claim1_text, claim_type, year}

# ── HELPER: STREAM-PARSE A CLAIMS TSV ─────────────────────────────────────────
def extract_claim1_from_tsv(tsv_stream, target_ids, id_field, year, claim_type):
    """
    Stream through a claims TSV and collect claim_sequence=1 rows
    for any patent_id in target_ids.
    Returns list of {patent_id, claim1_text, claim_type, year} dicts.
    """
    found = []
    seen_ids = set()
    chunk_size = 50_000

    # Read the stream in chunks
    reader = pd.read_csv(
        tsv_stream,
        sep="\t",
        dtype=str,
        chunksize=chunk_size,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace"
    )

    for chunk in reader:
        chunk.columns = [c.strip() for c in chunk.columns]

        # Verify required columns exist
        if id_field not in chunk.columns:
            # Try to find the ID column by inspection
            possible = [c for c in chunk.columns if "patent" in c.lower() or "doc" in c.lower() or "id" in c.lower()]
            if possible:
                id_field = possible[0]
                print(f"    Note: using '{id_field}' as ID column")
            else:
                print(f"    Error: could not find ID column. Columns: {list(chunk.columns)}")
                break

        seq_col = next((c for c in chunk.columns if "sequence" in c.lower()), None)
        text_col = next((c for c in chunk.columns if "text" in c.lower() or "claim_text" in c.lower()), None)

        if not seq_col or not text_col:
            print(f"    Error: missing sequence/text columns. Columns: {list(chunk.columns)}")
            break

        # Filter to first independent claim.
        # Granted files use 0-based indexing (sequence "0" = claim 1).
        # Pre-grant files use 1-based indexing (sequence "1" = claim 1).
        # Most robust: match whichever sequence value yields claim text starting with "1. "
        seq0 = chunk[chunk[seq_col].str.strip() == "0"]
        seq1 = chunk[chunk[seq_col].str.strip() == "1"]
        # Use seq0 if it has claim text starting "1. ", else fall back to seq1
        if len(seq0) > 0 and seq0[text_col].str.strip().str.startswith("1.").any():
            claim1 = seq0
        else:
            claim1 = seq1

        # Filter to target patents
        claim1 = claim1[claim1[id_field].isin(target_ids)]

        # Deduplicate (in case of reruns or duplicates in source)
        claim1 = claim1[~claim1[id_field].isin(seen_ids)]

        for _, row in claim1.iterrows():
            pid = row[id_field].strip()
            seen_ids.add(pid)
            found.append({
                "doc_id":       reissue_numeric.get(pid, pid),  # restore RE prefix if reissue
                "claim1_text":  str(row[text_col]).strip(),
                "claim_type":   claim_type,
                "year":         year
            })

    return found


def download_and_extract(url, zip_path):
    """Download a zip file and return the inner TSV as a file-like object."""
    if not os.path.exists(zip_path):
        print(f"    Downloading {url} ...")
        resp = requests.get(url, stream=True, timeout=120)
        if resp.status_code == 404:
            return None, f"404 Not Found: {url}"
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"    {downloaded/1e6:.0f} MB / {total/1e6:.0f} MB ({pct:.0f}%)  ", end="\r")
        print()
    else:
        print(f"    Already downloaded: {zip_path}")

    # Open zip and return first TSV file stream
    try:
        zf = zipfile.ZipFile(zip_path)
        tsv_names = [n for n in zf.namelist() if n.endswith(".tsv")]
        if not tsv_names:
            return None, f"No TSV found in zip: {zip_path}"
        return io.TextIOWrapper(zf.open(tsv_names[0]), encoding="utf-8", errors="replace"), None
    except zipfile.BadZipFile as e:
        return None, f"Bad zip file {zip_path}: {e}"


# ── STEP 2: PROCESS GRANTED CLAIMS ────────────────────────────────────────────
# ── RESUME LOGIC ─────────────────────────────────────────────────────────────
# If claims_claim1.csv already exists, load prior results and skip granted phase
RESUME = os.path.exists(OUTPUT_CSV)
if RESUME:
    print(f"\nResume mode: loading existing results from {OUTPUT_CSV}")
    prior = pd.read_csv(OUTPUT_CSV, dtype=str)
    results = prior.to_dict("records")
    print(f"  Loaded {len(results)} existing records — skipping granted phase")
else:
    results = []

print(f"\nProcessing GRANTED claims (g_claims) for {len(YEARS)} years...")
granted_found = 0

for year in YEARS:
    if RESUME:
        print(f"  Year {year}: SKIPPED (resume mode)")
        continue

    print(f"\n  Year {year}:")
    url      = GRANTED_URL.format(year=year)
    zip_path = os.path.join(DOWNLOAD_DIR, f"g_claims_{year}.tsv.zip")

    tsv_stream, err = download_and_extract(url, zip_path)
    if err:
        print(f"    WARN: {err} — skipping year {year} granted")
        if not KEEP_ZIPS and os.path.exists(zip_path):
            os.remove(zip_path)
        continue

    year_results = extract_claim1_from_tsv(
        tsv_stream,
        target_ids  = granted_ids_for_join,
        id_field    = "patent_id",
        year        = year,
        claim_type  = "granted"
    )
    results.extend(year_results)
    granted_found += len(year_results)
    print(f"    Found {len(year_results)} claim-1 records  (total so far: {granted_found})")

    if not KEEP_ZIPS:
        os.remove(zip_path)


# ── STEP 3: PROCESS PRE-GRANT CLAIMS ─────────────────────────────────────────
print(f"\nProcessing PRE-GRANT claims (pg_claims) for {len(YEARS)} years...")
pregrant_found = 0

for year in YEARS:
    print(f"\n  Year {year}:")
    url      = PREGRANT_URL.format(year=year)
    zip_path = os.path.join(DOWNLOAD_DIR, f"pg_claims_{year}.tsv.zip")

    tsv_stream, err = download_and_extract(url, zip_path)
    if err:
        print(f"    WARN: {err} — skipping year {year} pre-grant")
        if not KEEP_ZIPS and os.path.exists(zip_path):
            os.remove(zip_path)
        continue

    year_results = extract_claim1_from_tsv(
        tsv_stream,
        target_ids  = pregrant_ids,
        id_field    = "document_number",    # PatentsView pre-grant ID field
        year        = year,
        claim_type  = "pregrant"
    )
    results.extend(year_results)
    pregrant_found += len(year_results)
    print(f"    Found {len(year_results)} claim-1 records  (total so far: {pregrant_found})")

    if not KEEP_ZIPS:
        os.remove(zip_path)


# ── STEP 4: SAVE OUTPUT ────────────────────────────────────────────────────────
print(f"\nSaving output...")
claims_df = pd.DataFrame(results)

if len(claims_df) == 0:
    print("ERROR: No claims found. Check download URLs and PatentsView status.")
else:
    # Deduplicate — keep first occurrence per doc_id
    claims_df = claims_df.drop_duplicates(subset=["doc_id"], keep="first")
    claims_df.to_csv(OUTPUT_CSV, index=False)

    total_universe = len(granted_ids) + len(pregrant_ids) + len(reissue_ids)
    coverage = len(claims_df) / total_universe * 100

    print(f"\n── SUMMARY ───────────────────────────────────────────────────")
    print(f"  Granted claim-1 records  : {granted_found:>10,}")
    print(f"  Pre-grant claim-1 records: {pregrant_found:>10,}")
    print(f"  Total saved              : {len(claims_df):>10,}")
    print(f"  Coverage of AIPD universe: {coverage:.1f}%")
    print(f"  Output → {OUTPUT_CSV}")
    print(f"\n  Next step: run merge_claims.py to join claim1_text into")
    print(f"  the LLM input pipeline alongside title + abstract.")