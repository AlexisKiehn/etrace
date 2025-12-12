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


def upload_to_gcs(bucket_name: str, destination_blob_name: str, source_file_name: str):
    """Uploads a file to Google Cloud Storage."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}."
    )


if __name__ == "__main__":
    # Example usage
    upload_to_gcs(
        "etrace-data",
        "data/koppen_tif_files/",
        "/Users/ak/code/AlexisKiehn/etrace/koppen_tif_files/koppen_1901_1930.tif",
    )
