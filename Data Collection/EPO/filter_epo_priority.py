"""
filter_epo_priority.py
Fetches earliest priority country for each EP patent via OPS API
and filters out patents where first priority was filed at USPTO (US).

Input:  ~/Downloads/epo_candidates_v3.csv
Output: ~/Downloads/epo_candidates_v4.csv  (European-origin only)

Usage:
    export EPO_CLIENT_ID="your_key"
    export EPO_CLIENT_SECRET="your_secret"
    /Users/swayamchadha/miniconda3/bin/python3 filter_epo_priority.py
"""

import requests, time, os, re, csv
import pandas as pd

BASE_URL   = "https://ops.epo.org/3.2/rest-services"
INPUT_CSV  = os.path.expanduser("~/Downloads/epo_candidates_v3.csv")
OUTPUT_CSV = os.path.expanduser("~/Downloads/epo_candidates_v4.csv")
CACHE_CSV  = os.path.expanduser("~/Downloads/epo_priority_cache.csv")

CID  = os.environ.get("EPO_CLIENT_ID", "")
CSEC = os.environ.get("EPO_CLIENT_SECRET", "")
_token = {"value": None}


def get_token():
    if not CID:
        return None
    resp = requests.post(
        "https://ops.epo.org/3.2/auth/accesstoken",
        data={"grant_type": "client_credentials"},
        auth=(CID, CSEC), timeout=15
    )
    if resp.status_code == 200:
        _token["value"] = resp.json().get("access_token")
        print("Token refreshed OK")
        return _token["value"]
    return None


def make_headers():
    h = {"Accept": "application/json"}
    if _token["value"]:
        h["Authorization"] = f"Bearer {_token['value']}"
    return h


def api_get(url, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=make_headers(), timeout=30)
        except requests.exceptions.ReadTimeout:
            print(f"  Timeout attempt {attempt+1} — retrying...")
            time.sleep(10)
            continue
        except requests.exceptions.ConnectionError:
            time.sleep(30)
            continue

        if resp.status_code in (403, 429):
            print(f"  Rate limited — sleeping 60s...")
            time.sleep(60)
            continue
        if resp.status_code == 400 and "invalid_access_token" in resp.text:
            print(f"  Token expired — refreshing...")
            get_token()
            continue
        return resp

    class FakeResp:
        status_code = 500
        text = "max retries"
        def json(self): return {}
    return FakeResp()


def get_priority_country(doc_id):
    """Returns earliest priority country code, or None if unavailable."""
    m = re.match(r"([A-Z]{2})(\d+)([A-Z]\d?)?", doc_id)
    if not m:
        return None
    cc, num, kind = m.group(1), m.group(2), m.group(3) or "A1"

    resp = api_get(
        f"{BASE_URL}/published-data/publication/epodoc/{cc}.{num}.{kind}/biblio"
    )
    if resp.status_code != 200:
        return None

    try:
        xd = resp.json()["ops:world-patent-data"]["exchange-documents"]["exchange-document"]
        if isinstance(xd, list):
            xd = xd[0]

        pri_claims = xd.get("bibliographic-data", {}).get("priority-claims", {})
        priorities  = pri_claims.get("priority-claim", [])
        if isinstance(priorities, dict):
            priorities = [priorities]

        # Find earliest priority by sequence number
        if not priorities:
            return None

        # Sort by sequence
        def seq(p):
            try:
                return int(p.get("@sequence", 99))
            except Exception:
                return 99

        priorities = sorted(priorities, key=seq)
        first = priorities[0]

        # doc-number format is like "EP20190305580" or "US20180123456"
        # country code is the first 2 characters
        did = first.get("document-id", {})
        if isinstance(did, list):
            did = did[0]
        doc_num = did.get("doc-number", {}).get("$", "")
        # Extract leading 2-letter country code
        import re as _re
        m = _re.match(r"([A-Z]{2})", doc_num)
        if m:
            return m.group(1)
        # Fallback: try country field directly
        country = did.get("country", {}).get("$", "")
        return country if country else None
    except Exception:
        return None


def main():
    get_token()

    df = pd.read_csv(INPUT_CSV)
    df = df[df['ep_id'].str.startswith('EP')].copy()
    print(f"Total EP patents: {len(df)}")

    # Load cache (delete epo_priority_cache.csv first if re-running after parser fix)
    cache = {}
    if os.path.exists(CACHE_CSV):
        cache_df = pd.read_csv(CACHE_CSV, dtype=str)
        # Only load entries that are not "unknown"
        cache_df = cache_df[cache_df["priority_country"] != "unknown"]
        cache = dict(zip(cache_df["ep_id"], cache_df["priority_country"]))
        print(f"Loaded {len(cache)} cached priority countries (skipping unknown)")

    # Fetch priority countries
    new_cache = dict(cache)
    ids_to_fetch = [ep_id for ep_id in df['ep_id'].tolist() if ep_id not in cache]
    print(f"Need to fetch priority for {len(ids_to_fetch)} patents...")

    for i, ep_id in enumerate(ids_to_fetch):
        country = get_priority_country(ep_id)
        new_cache[ep_id] = country or "unknown"

        if (i + 1) % 100 == 0:
            # Save cache checkpoint
            cache_df = pd.DataFrame([
                {'ep_id': k, 'priority_country': v}
                for k, v in new_cache.items()
            ])
            cache_df.to_csv(CACHE_CSV, index=False)
            print(f"  [{i+1}/{len(ids_to_fetch)}] cached")
        time.sleep(0.5)

    # Final cache save
    cache_df = pd.DataFrame([
        {'ep_id': k, 'priority_country': v}
        for k, v in new_cache.items()
    ])
    cache_df.to_csv(CACHE_CSV, index=False)

    # Apply filter
    df['priority_country'] = df['ep_id'].map(new_cache)
    print(f"\nPriority country distribution:")
    print(df['priority_country'].value_counts().head(15).to_string())

    # Exclude US-priority patents
    non_us = df[df['priority_country'] != 'US'].copy()
    print(f"\nTotal: {len(df)}, Excluding US-priority: {len(df) - len(non_us)}")
    print(f"Remaining European-origin patents: {len(non_us)}")

    # Also filter to patents with abstracts
    non_us = non_us[non_us['abstract'].notna() & (non_us['abstract'] != '')].copy()
    non_us = non_us.drop_duplicates(subset='ep_id').copy()
    print(f"With abstracts, deduplicated: {len(non_us)}")
    print(f"\nBy year:")
    print(non_us.groupby('year').size().to_string())

    non_us.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()