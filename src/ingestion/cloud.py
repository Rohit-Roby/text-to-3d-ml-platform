from __future__ import annotations

from google.cloud import bigquery, storage


def upload_file_to_gcs(local_path: str, bucket_name: str, blob_name: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_name).upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"


def load_parquet_to_bigquery(parquet_uri: str, table_id: str) -> None:
    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET, write_disposition="WRITE_TRUNCATE"
    )
    client.load_table_from_uri(parquet_uri, table_id, job_config=job_config).result()
