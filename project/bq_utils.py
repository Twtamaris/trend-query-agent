from google.cloud import bigquery
from datetime import datetime, timedelta
from config import PROJECT_ID, DATASET_ID, CONVERSATIONS_TABLE, DEFAULT_BQ_START_DATE, DEFAULT_BQ_END_DATE

# Initialize BigQuery client
client = bigquery.Client()

def get_conversation_data(zip_code=None, start_date=None, end_date=None, sentiment=None, issue_category=None):
    """
    Retrieves raw conversation data (without embeddings) from BigQuery based on filters.
    Dates should be in 'YYYY-MM-DD' format.
    Queries the 'conversations' table.
    """
    query = f"""
    SELECT
        conversation_id,
        conversation_text,
        sentiment,
        issue_category,
        timestamp,
        zip_code
    FROM
        `{PROJECT_ID}.{DATASET_ID}.{CONVERSATIONS_TABLE}`
    WHERE
        1=1
    """

    # Track if any date filter was applied by arguments
    date_filter_applied = False
    if start_date:
        query += f" AND DATE(timestamp) >= DATE('{start_date}')"
        date_filter_applied = True
    if end_date:
        query += f" AND DATE(timestamp) <= DATE('{end_date}')"
        date_filter_applied = True

    # If no date filters were explicitly provided by the arguments, apply a broad default.
    # This is crucial for partitioned tables.
    if not date_filter_applied:
        query += f" AND DATE(timestamp) >= DATE('{DEFAULT_BQ_START_DATE}') AND DATE(timestamp) <= DATE('{DEFAULT_BQ_END_DATE}')"
        print(f"Applying default date filter from {DEFAULT_BQ_START_DATE} to {DEFAULT_BQ_END_DATE} for partitioned table.")


    if zip_code:
        query += f" AND zip_code = '{zip_code}'"
    if sentiment:
        query += f" AND LOWER(sentiment) = LOWER('{sentiment}')"
    if issue_category:
        query += f" AND LOWER(issue_category) = LOWER('{issue_category}')"

    print(f"Executing BigQuery query for filtering...\n{query}") # Uncomment for debugging
    query_job = client.query(query)
    results = query_job.result()

    processed_results = []
    for row in results:
        row_dict = dict(row)
        # Ensure conversation_id is a string, useful for lookups later
        row_dict['conversation_id'] = str(row_dict['conversation_id'])
        processed_results.append(row_dict)

    print(f"Retrieved {len(processed_results)} rows from BigQuery after filtering.")
    return processed_results