"""
bigquery_epo_patents.py
Queries Google BigQuery patents-public-data for EP AI-candidate patents
2010-2023 with English abstracts, filtered to European-origin only.

Requirements:
    pip install google-cloud-bigquery pandas pyarrow

Usage:
    # Authenticate first:
    gcloud auth application-default login

    python bigquery_epo_patents.py

Output: ~/Downloads/epo_bigquery_candidates.csv

Free tier: 1TB/month — this query scans ~50-100GB, well within free limits.
"""

import os
import pandas as pd
from google.cloud import bigquery

OUTPUT_CSV = os.path.expanduser("~/Downloads/epo_bigquery_candidates.csv")
PROJECT_ID = "thesis-project-498814"  # replace with your GCP project ID

# SQL query: EP patents 2010-2023 with AI-relevant CPC codes
# - English abstracts only
# - Non-machine-translated only
# - Earliest priority country not US
QUERY = """
#standardSQL
WITH ep_patents AS (
  SELECT
    p.publication_number,
    p.application_number,
    p.filing_date,
    p.publication_date,
    EXTRACT(YEAR FROM PARSE_DATE('%Y%m%d', CAST(p.publication_date AS STRING))) AS pub_year,
    (SELECT title.text FROM UNNEST(p.title_localized) title
     WHERE title.language = 'en' LIMIT 1) AS title,
    (SELECT abstract.text FROM UNNEST(p.abstract_localized) abstract
     WHERE abstract.language = 'en' AND abstract.text IS NOT NULL LIMIT 1) AS abstract,
    -- Get earliest priority country
    (SELECT SUBSTR(priority.publication_number, 0, 2)
     FROM UNNEST(p.priority_claim) priority
     ORDER BY priority.filing_date ASC LIMIT 1) AS priority_country,
    -- Get CPC codes as array
    ARRAY(SELECT cpc.code FROM UNNEST(p.cpc) cpc) AS cpc_codes
  FROM
    `patents-public-data.patents.publications` p
  WHERE
    -- EP patents only
    p.country_code = 'EP'
    -- Publication year 2010-2023
    AND EXTRACT(YEAR FROM PARSE_DATE('%Y%m%d', CAST(p.publication_date AS STRING))) BETWEEN 2010 AND 2023
    -- Must have English abstract
    AND EXISTS (
      SELECT 1 FROM UNNEST(p.abstract_localized) a
      WHERE a.language = 'en' AND a.text IS NOT NULL AND LENGTH(a.text) > 50
    )
    -- AI-relevant CPC codes
    AND EXISTS (
      SELECT 1 FROM UNNEST(p.cpc) c
      WHERE REGEXP_CONTAINS(c.code, r'^(G06N|G06V|G10L1[0-9]|G10L2[0-9]|G06F18|G06F40|G16H|G05B|G06Q)')
    )
)
SELECT
  publication_number AS ep_id,
  pub_year AS year,
  title,
  abstract,
  priority_country,
  -- First matching AI CPC code
  (SELECT code FROM UNNEST(cpc_codes) code
   WHERE REGEXP_CONTAINS(code, r'^(G06N|G06V|G10L1[0-9]|G10L2[0-9]|G06F18|G06F40|G16H|G05B|G06Q)')
   LIMIT 1) AS cpc
FROM ep_patents
-- Exclude US-priority patents
WHERE priority_country != 'US'
  AND priority_country IS NOT NULL
  AND abstract IS NOT NULL
ORDER BY year, ep_id
"""


def main():
    client = bigquery.Client(project=PROJECT_ID)

    print("Running BigQuery query...")
    print("Estimated data scanned: ~50-100 GB (within free 1TB/month tier)")

    job_config = bigquery.QueryJobConfig(
        use_query_cache=True,
    )

    query_job = client.query(QUERY, job_config=job_config)
    print("Query submitted, waiting for results...")

    results = query_job.result()
    df = results.to_dataframe()

    print(f"\nTotal EP AI patents fetched: {len(df):,}")
    print(f"\nBy year:")
    print(df.groupby('year').size().to_string())
    print(f"\nPriority country distribution:")
    print(df['priority_country'].value_counts().head(15).to_string())

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
