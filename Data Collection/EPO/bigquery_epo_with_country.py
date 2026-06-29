from google.cloud import bigquery
import pandas as pd

client = bigquery.Client(project="")

# The existing EP IDs from the classification run
# We fetch applicant country from the assignees/inventors table
# Priority country = country of the first/earliest priority filing
# Applicant country = country of the first applicant listed

QUERY = """
WITH ep_patents AS (
  SELECT
    p.publication_number,
    CAST(FLOOR(p.publication_date / 10000) AS INT64) AS year,
    -- Applicant country: country of first assignee (harmonized)
    (SELECT a.country_code
     FROM UNNEST(p.assignee_harmonized) AS a
     LIMIT 1) AS applicant_country,
    -- Inventor country: country of first inventor (harmonized)
    (SELECT i.country_code
     FROM UNNEST(p.inventor_harmonized) AS i
     LIMIT 1) AS inventor_country
  FROM
    `patents-public-data.patents.publications` AS p
  WHERE
    REGEXP_CONTAINS(p.publication_number, r'^EP')
    AND EXISTS (
      SELECT 1 FROM UNNEST(p.abstract_localized) AS ab
      WHERE ab.language = 'en' AND LENGTH(ab.text) > 50
    )
    AND EXISTS (
      SELECT 1 FROM UNNEST(p.cpc) AS c
      WHERE REGEXP_CONTAINS(c.code, r'^(G06N|G06V|G06F|G16H|G05B|G06Q|G10L)')
    )
    AND CAST(FLOOR(p.publication_date / 10000) AS INT64) BETWEEN 2010 AND 2023
    AND NOT EXISTS (
      SELECT 1 FROM UNNEST(p.priority_claim) AS pr
      WHERE pr.application_number LIKE 'US%'
    )
)
SELECT
  publication_number AS ep_id,
  year,
  applicant_country,
  inventor_country
FROM ep_patents
WHERE year BETWEEN 2010 AND 2023
"""

print("Running BigQuery query to fetch EPO patents with country information...")
print("This may take 1-2 minutes...")

df = client.query(QUERY).to_dataframe()
print(f"Fetched {len(df):,} rows")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nApplicant country distribution (top 20):")
print(df['applicant_country'].value_counts().head(20))
print(f"\nInventor country distribution (top 20):")
print(df['inventor_country'].value_counts().head(20))

df.to_csv('epo_candidates_with_country.csv', index=False)
print(f"\nSaved to epo_candidates_with_country.csv")
