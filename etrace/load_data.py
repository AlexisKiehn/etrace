from google.cloud import bigquery

client = bigquery.Client()

def load_from_bq(query: str):
    query_job = client.query(query)
    results = query_job.result()
    return results.to_dataframe()
