from google.cloud.storage import Blob, Client
from google.cloud import bigquery

import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

client = bigquery.Client(credentials=credentials)


def load_from_bq(query: str):
    query_job = client.query(query)
    results = query_job.result()
    return results.to_dataframe()


def load_from_bucket(
    bucket_name: str, source_blob_name: str, destination_file_name: str
):
    """Downloads a file from Google Cloud Storage."""
    storage_client = Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(
        f"File {source_blob_name} downloaded to {destination_file_name} from bucket {bucket_name}."
    )

    return destination_file_name


if __name__ == "__main__":

    load_from_bucket(
        "etrace-data",
        "data/area.ipynb",
        "/Users/ak/code/AlexisKiehn/etrace/area_downloaded.ipynb",
    )
