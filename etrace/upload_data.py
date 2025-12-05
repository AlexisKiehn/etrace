from google.cloud.storage import Blob, Client

client = Client()


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
        "etrace-data", "data/area.ipynb", "/Users/ak/code/AlexisKiehn/etrace/area.ipynb"
    )
