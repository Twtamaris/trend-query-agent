from google.cloud import bigquery
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load .env (make sure GOOGLE_APPLICATION_CREDENTIALS is set here or in your env)
load_dotenv()

# Initialize BigQuery client
client = bigquery.Client()

print("Loading Sentence Transformer model 'all-MiniLM-L6-v2'...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.\n")

def generate_and_store_embeddings_in_bq(
    project_id: str,
    dataset_id: str,
    source_table_id: str,
    destination_table_id: str
):
    src = f"`{project_id}.{dataset_id}.{source_table_id}`"
    dst = f"{project_id}.{dataset_id}.{destination_table_id}"

    print(f"Fetching data from {src} for embedding generation...")
    query = f"""
      SELECT *
      FROM {src}
      WHERE timestamp
        BETWEEN DATETIME('1970-01-01 00:00:00')
            AND CURRENT_DATETIME()
    """
    try:
        rows = list(client.query(query).result())
    except Exception as e:
        print(f"Error fetching data from BigQuery: {e}")
        return

    if not rows:
        print("No rows in the specified date range; exiting.")
        return

    df = pd.DataFrame([dict(r) for r in rows])
    print(f" → Retrieved {len(df)} rows.")

    if 'conversation_text' not in df:
        print("Error: no 'conversation_text' column.")
        return

    # Prepare texts & embeddings
    df['conversation_text'] = df['conversation_text'].fillna('')
    texts = df['conversation_text'].tolist()
    print(f"Generating embeddings for {len(texts)} records...")
    embs = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    df['embedding'] = [e.tolist() for e in embs]
    print(" → Embeddings generated.")

    # Build BigQuery schema
    schema = []
    for col, dt in df.dtypes.items():
        if col == 'embedding':
            schema.append(
                bigquery.SchemaField(
                    name=col,
                    field_type="FLOAT",
                    mode="REPEATED",
                    description="Sentence embedding"
                )
            )
        elif col == 'timestamp':
            schema.append(
                bigquery.SchemaField(
                    name=col,
                    field_type="TIMESTAMP",
                    mode="NULLABLE"
                )
            )
        elif 'int' in str(dt):
            schema.append(bigquery.SchemaField(col, "INTEGER", mode="NULLABLE"))
        elif 'float' in str(dt):
            schema.append(bigquery.SchemaField(col, "FLOAT", mode="NULLABLE"))
        else:
            schema.append(bigquery.SchemaField(col, "STRING", mode="NULLABLE"))

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_TRUNCATE",
        time_partitioning=bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="timestamp"
        ),
        clustering_fields=["zip_code", "issue_category"]
    )

    # Ensure timestamp column is actual Python datetime or pandas Timestamp
    if df['timestamp'].dtype == object:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Uploading to {dst} ...")
    try:
        load_job = client.load_table_from_dataframe(
            dataframe=df,
            destination=dst,
            job_config=job_config
        )
        load_job.result()  # wait for completion
        print(" → Upload complete!")
    except Exception as e:
        print(f"Error during upload: {e}")

if __name__ == "__main__":
    generate_and_store_embeddings_in_bq(
        project_id='chatbot-project-464108',
        dataset_id='customer_service_data',
        source_table_id='conversations',
        destination_table_id='conversations_with_embeddings'
    )
    print("\nDone — check your new table in the BigQuery console.")




# # import os
# from google.cloud import bigquery
# import pandas as pd
# import numpy as np # For handling numpy arrays
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv # Assuming you use dotenv for environment variables

# # Load environment variables from .env file (if you have one)
# load_dotenv()

# # Initialize BigQuery client
# client = bigquery.Client()

# # --- Load a pre-trained embedding model ---
# # 'all-MiniLM-L6-v2' is a good general-purpose model
# print("Loading Sentence Transformer model 'all-MiniLM-L6-v2'...")
# model = SentenceTransformer('all-MiniLM-L6-v2')
# print("Model loaded.")

# def generate_and_store_embeddings_in_bq(project_id, dataset_id, source_table_id, destination_table_id):
#     """
#     Fetches conversation texts from a BigQuery table, generates embeddings,
#     and creates a new BigQuery table with an added 'embedding' column.

#     Args:
#         project_id (str): Your Google Cloud Project ID.
#         dataset_id (str): The BigQuery dataset ID (e.g., 'customer_service_data').
#         source_table_id (str): The ID of the table containing conversation text (e.g., 'conversations').
#         destination_table_id (str): The ID for the new table where data with embeddings will be stored.
#                                     (e.g., 'conversations_with_embeddings' or 'conversations_reloaded')
#     """
#     source_table_ref = f"`{project_id}.{dataset_id}.{source_table_id}`"
#     destination_table_full_id = f"{project_id}.{dataset_id}.{destination_table_id}"

#     # 1. Fetch conversation data from BigQuery
#     print(f"Fetching data from {source_table_ref} for embedding generation...")
#     query = f"SELECT * FROM {source_table_ref}" # Select all columns
#     query_job = client.query(query)
#     rows = list(query_job.result())

#     if not rows:
#         print(f"No data found in {source_table_ref}. Exiting.")
#         return

#     # Convert to DataFrame
#     df = pd.DataFrame([dict(row) for row in rows])
#     print(f"Fetched {len(df)} rows from {source_table_id}.")

#     # Ensure conversation_text column exists and handle potential None values
#     if 'conversation_text' not in df.columns:
#         print("Error: 'conversation_text' column not found in the source table. Please check your schema.")
#         return
    
#     # Fill NaN/None values in conversation_text with empty string before encoding
#     df['conversation_text'] = df['conversation_text'].fillna('')
#     texts_to_encode = df['conversation_text'].tolist()

#     # 2. Generate Embeddings
#     print(f"Generating embeddings for {len(texts_to_encode)} conversations...")
#     embeddings = model.encode(texts_to_encode, show_progress_bar=True, normalize_embeddings=True)
#     # normalize_embeddings=True is important for cosine similarity to work correctly

#     # Convert embeddings (numpy arrays) to Python lists for BigQuery ARRAY<FLOAT64> type
#     df['embedding'] = [emb.tolist() for emb in embeddings]
#     print("Embeddings generated and added to DataFrame.")

#     # 3. Prepare Schema for the New Table
#     # Get existing schema fields and add the new 'embedding' field
#     # You might need to adjust this part if your original table schema has specific modes (REQUIRED, NULLABLE)
#     # For simplicity, we'll try to infer and then add.
#     original_schema = client.get_table(source_table_ref.strip('`')).schema
#     new_schema_fields = list(original_schema) # Start with original fields

#     # Check if 'embedding' already exists to avoid adding it multiple times on reruns
#     if 'embedding' not in [field.name for field in new_schema_fields]:
#         new_schema_fields.append(bigquery.SchemaField("embedding", "ARRAY", mode="NULLABLE", fields=[
#             bigquery.SchemaField("float_value", "FLOAT", mode="NULLABLE") # BigQuery ARRAY<FLOAT64> doesn't need sub-fields if not struct
#         ]))
#         # For a simple ARRAY<FLOAT64>, you just need:
#         # new_schema_fields.append(bigquery.SchemaField("embedding", "ARRAY", mode="NULLABLE", field_type="FLOAT"))
#         # This seems to be a common point of confusion for `ARRAY<FLOAT64>` in `bigquery.SchemaField`.
#         # Let's use a simpler form:
#         new_schema_fields = []
#         for col in df.columns:
#             if col == 'embedding':
#                 # For ARRAY<FLOAT64>, BigQuery expects mode="REPEATED" for the field,
#                 # and then the type inside the list.
#                 # However, the BigQuery client library expects type="ARRAY" and then
#                 # the "fields" argument can specify the element type directly for scalar arrays.
#                 # Let's try the direct type specification for the array elements:
#                 new_schema_fields.append(bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED"))
#             elif col == 'timestamp': # Ensure timestamp is handled correctly, potentially as STRING for upload
#                 new_schema_fields.append(bigquery.SchemaField(col, "TIMESTAMP", mode="NULLABLE"))
#             elif col == 'zip_code': # Ensure zip_code is handled as STRING
#                  new_schema_fields.append(bigquery.SchemaField(col, "STRING", mode="NULLABLE"))
#             else:
#                 # Infer type for other columns from DataFrame dtypes, or explicitly map
#                 # For robust solution, match original BigQuery schema or provide explicit mappings
#                 # For now, let's assume they are strings or other basic types that pandas handles well.
#                 # A more robust solution would iterate through original_schema to preserve types.
#                 # For this example, let's just make everything else STRING for simplicity or use pandas' infer_dtypes
#                 dtype = str(df[col].dtype)
#                 if 'int' in dtype:
#                     new_schema_fields.append(bigquery.SchemaField(col, "INTEGER", mode="NULLABLE"))
#                 elif 'float' in dtype:
#                     new_schema_fields.append(bigquery.SchemaField(col, "FLOAT", mode="NULLABLE"))
#                 elif 'bool' in dtype:
#                     new_schema_fields.append(bigquery.SchemaField(col, "BOOLEAN", mode="NULLABLE"))
#                 else:
#                     new_schema_fields.append(bigquery.SchemaField(col, "STRING", mode="NULLABLE"))

#     # To be precise, let's rebuild the schema based on the DataFrame columns
#     # and map to BigQuery types. This is more reliable.
#     bq_schema_for_upload = []
#     for column_name, dtype in df.dtypes.items():
#         if column_name == 'embedding':
#             bq_schema_for_upload.append(bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED"))
#         elif column_name == 'timestamp':
#             bq_schema_for_upload.append(bigquery.SchemaField("timestamp", "TIMESTAMP", mode="NULLABLE"))
#         elif column_name == 'zip_code':
#             bq_schema_for_upload.append(bigquery.SchemaField("zip_code", "STRING", mode="NULLABLE"))
#         elif 'int' in str(dtype):
#             bq_schema_for_upload.append(bigquery.SchemaField(column_name, "INTEGER", mode="NULLABLE"))
#         elif 'float' in str(dtype):
#             bq_schema_for_upload.append(bigquery.SchemaField(column_name, "FLOAT", mode="NULLABLE"))
#         elif 'bool' in str(dtype):
#             bq_schema_for_upload.append(bigquery.SchemaField(column_name, "BOOLEAN", mode="NULLABLE"))
#         else: # Default to STRING for anything else
#             bq_schema_for_upload.append(bigquery.SchemaField(column_name, "STRING", mode="NULLABLE"))


#     job_config = bigquery.LoadJobConfig(
#         schema=bq_schema_for_upload,
#         write_disposition="WRITE_TRUNCATE",  # Overwrite if table exists
#         time_partitioning=bigquery.TimePartitioning(
#             type_=bigquery.TimePartitioningType.DAY,
#             field="timestamp",  # Assuming 'timestamp' is your partitioning field
#             require_partition_filter=True,
#         ),
#         clustering_fields=["zip_code", "issue_category"], # Assuming these are still valid
#         # For ARRAY<FLOAT64>, BigQuery doesn't directly support `source_format="CSV"` for nested types
#         # You need to upload as JSON or directly from dataframe using `to_gbq` for example.
#     )

#     # 4. Upload DataFrame to New BigQuery Table
#     print(f"Uploading data with embeddings to {destination_table_full_id}...")
#     try:
#         # Using client.load_table_from_dataframe is often the easiest for DataFrames
#         # Ensure that pandas_gbq is installed: pip install pandas-gbq
#         from pandas_gbq import to_gbq
        
#         # Convert timestamp column to timezone-aware datetime objects if needed,
#         # or ensure it's in a format BigQuery expects for TIMESTAMP.
#         # df['timestamp'] = pd.to_datetime(df['timestamp']) # Uncomment if your timestamp is a string
#                                                          # and needs conversion for BigQuery to handle.
#         # However, it's already fetched as datetime objects by default from BigQuery,
#         # so direct conversion might not be needed if coming from query_job.result()
#         # but could be if your source was local CSV etc.

#         to_gbq(
#             df,
#             destination_table_full_id,
#             project_id=project_id,
#             if_exists='replace', # 'replace' will overwrite, 'append' will add
#             table_schema=bq_schema_for_upload # Pass the defined schema
#         )

#         print(f"Successfully created and loaded data into table: {destination_table_full_id}")
#         print("Note: The clustering and partitioning will be applied automatically based on the job_config settings if using client.load_table_from_dataframe correctly or recreating the table with these options.")

#     except Exception as e:
#         print(f"Error uploading data to BigQuery: {e}")
#         print("Please ensure 'pandas-gbq' is installed (`pip install pandas-gbq`)")
#         print("Also, check that your BigQuery service account has 'BigQuery Data Editor' role.")

# # --- Run this once to generate embeddings and store them ---
# if __name__ == "__main__":
#     # Replace with your actual project, dataset, and table IDs
#     project_id = 'chatbot-project-464108' # Your actual project ID
#     dataset_id = 'customer_service_data'
#     source_table_id = 'conversations' # Your original table name
#     # We'll create a new table to hold the embeddings
#     destination_table_id = 'conversations_with_embeddings' 

#     generate_and_store_embeddings_in_bq(project_id, dataset_id, source_table_id, destination_table_id)

#     print("\n--- Next Steps ---")
#     print("1. Verify the new table 'conversations_with_embeddings' in BigQuery UI.")
#     print("2. Check if the 'embedding' column (ARRAY<FLOAT64>) is present and populated.")
#     print("3. You can now use this table for semantic search/similarity.")
