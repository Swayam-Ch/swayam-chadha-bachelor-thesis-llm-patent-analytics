"""
fetch_epo_patents.py — EPO OPS API patent fetcher
Auto-refreshes token on expiry.

Usage:
    export EPO_CLIENT_ID="your_key"
    export EPO_CLIENT_SECRET="your_secret"
    /Users/swayamchadha/miniconda3/bin/python3 fetch_epo_patents.py
"""

import requests, time, csv, os, re

BASE_URL   = "https://ops.epo.org/3.2/rest-services"
OUTPUT_CSV = os.path.expanduser("~/Downloads/epo_candidates_v3.csv")
YEARS      = list(range(2010, 2024))
MAX_PER_YEAR_CPC = 500

AI_CPCS = [
    ("G06N",   "AI/ML core"),
    ("G06V",   "Computer vision"),
    ("G10L15", "Speech recognition"),
    ("G10L25", "Speech analysis AI"),
    ("G06F",   "General computing/software AI"),
    ("G16H",   "Healthcare informatics"),
    ("G05B",   "Control systems/robotics"),
    ("G06Q",   "Business/finance methods"),
]

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
    print(f"Token failed: {resp.status_code}")
    return None


def make_headers():
    h = {"Accept": "application/json"}
    if _token["value"]:
        h["Authorization"] = f"Bearer {_token['value']}"
    return h


def api_get(url, params=None, extra_headers=None, retries=3):
    """GET with automatic token refresh on 400 invalid_access_token."""
    for attempt in range(retries):
        h = make_headers()
        if extra_headers:
            h.update(extra_headers)
        try:
            resp = requests.get(url, params=params, headers=h, timeout=60)
        except requests.exceptions.ReadTimeout:
            print(f"  Timeout on attempt {attempt+1} — retrying after 10s...")
            time.sleep(10)
            continue
        except requests.exceptions.ConnectionError:
            print(f"  Connection error on attempt {attempt+1} — retrying after 30s...")
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

    # create a fake response-like object if all retries failed
    class FakeResp:
        status_code = 500
        text = "max retries exceeded"
        def json(self): return {}
    return FakeResp()


def search_page(cpc, year, start):
    query = f'cpc={cpc} AND pd within "{year}0101 {year}1231" AND pn=EP'
    end   = start + 99

    resp = api_get(
        f"{BASE_URL}/published-data/search",
        params={"q": query},
        extra_headers={"Range": f"{start}-{end}"}
    )

    if resp.status_code != 200:
        print(f"  Error {resp.status_code}: {resp.text[:150]}")
        return [], 0

    try:
        data  = resp.json()
        bs    = data["ops:world-patent-data"]["ops:biblio-search"]
        total = int(bs.get("@total-result-count", 0))
        refs  = bs.get("ops:search-result", {}).get("ops:publication-reference", [])
        if isinstance(refs, dict):
            refs = [refs]
        ids = []
        for r in refs:
            did  = r.get("document-id", {})
            cc   = did.get("country",    {}).get("$", "")
            num  = did.get("doc-number", {}).get("$", "")
            kind = did.get("kind",       {}).get("$", "")
            if cc and num:
                ids.append(f"{cc}{num}{kind}")
        return ids, total
    except Exception as e:
        print(f"  Parse error: {e}")
        return [], 0


def fetch_abstract(doc_id):
    m = re.match(r"([A-Z]{2})(\d+)([A-Z]\d?)?", doc_id)
    if not m:
        return None, None
    cc, num, kind = m.group(1), m.group(2), m.group(3) or "A1"

    resp = api_get(
        f"{BASE_URL}/published-data/publication/epodoc/{cc}.{num}.{kind}/abstract"
    )
    if resp.status_code != 200:
        return None, None

    try:
        xd = resp.json()["ops:world-patent-data"]["exchange-documents"]["exchange-document"]
        if isinstance(xd, list):
            xd = xd[0]

        title = None
        titles = xd.get("bibliographic-data", {}).get("invention-title", [])
        if isinstance(titles, dict): titles = [titles]
        for t in titles:
            if isinstance(t, dict) and t.get("@lang") == "en":
                title = t.get("$", "")
                break

        abstract = None
        abs_data = xd.get("abstract", [])
        if isinstance(abs_data, dict): abs_data = [abs_data]
        for a in abs_data:
            if a.get("@lang") == "en":
                p = a.get("p", "")
                if isinstance(p, list):
                    abstract = " ".join(x.get("$", "") if isinstance(x, dict) else str(x) for x in p)
                elif isinstance(p, dict):
                    abstract = p.get("$", "")
                else:
                    abstract = str(p)
                break
        return title, abstract
    except Exception:
        return None, None


def save(results):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ep_id","year","cpc","title","abstract"])
        w.writeheader()
        w.writerows(results)


def main():
    get_token()
    results = []
    seen    = set()

    # Resume from checkpoint if exists (only if contains EP patents)
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["ep_id"].startswith("EP"):
                    seen.add(row["ep_id"])
                    results.append(row)
        print(f"Resuming from checkpoint: {len(results)} EP patents already fetched")

    for year in YEARS:
        for cpc, label in AI_CPCS:
            # Skip if already fully fetched this year/CPC
            existing = sum(1 for r in results if r["year"] == str(year) and r["cpc"] == cpc)
            if existing >= MAX_PER_YEAR_CPC:
                print(f"Year {year} / {cpc}: already have {existing}, skipping")
                continue

            print(f"\nYear {year}, CPC {cpc} ({label})...")
            start   = existing * 100 // 100 * 100 + 1  # approximate resume point
            fetched = existing

            while fetched < MAX_PER_YEAR_CPC:
                ids, total = search_page(cpc, year, start)
                if not ids:
                    break
                print(f"  [{start}-{start+len(ids)-1}] of {total} total")

                for doc_id in ids:
                    if doc_id in seen:
                        continue
                    seen.add(doc_id)
                    title, abstract = fetch_abstract(doc_id)
                    results.append({
                        "ep_id":    doc_id,
                        "year":     str(year),
                        "cpc":      cpc,
                        "title":    title    or "",
                        "abstract": abstract or "",
                    })
                    fetched += 1
                    time.sleep(0.3)

                if start + 99 >= total or fetched >= MAX_PER_YEAR_CPC:
                    break
                start += 100
                time.sleep(0.5)

            print(f"  -> {fetched} patents for {year}/{cpc}")

        # Checkpoint after each year
        save(results)
        print(f"  Checkpoint: {len(results)} total saved")

    print(f"\nDone. {len(results)} EP patents saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()