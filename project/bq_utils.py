from google.cloud import bigquery
from datetime import datetime, timedelta
from config import PROJECT_ID, DATASET_ID, CONVERSATIONS_TABLE, DEFAULT_BQ_START_DATE, DEFAULT_BQ_END_DATE
from datetime import datetime, timedelta, date # Import date as well

# Initialize BigQuery client
client = bigquery.Client()


# Helper function to safely get and format datetime/date fields
def get_and_format_date(data_dict, key):
    dt_obj = data_dict.get(key)
    if isinstance(dt_obj, (datetime, date)): # Check for both datetime and date objects
        return dt_obj.isoformat()
    # If it's a string that might be a date, you could try parsing it here
    # For now, if it's not a datetime object, just return it as is or None
    return dt_obj # Return as is, or None if you prefer strict control over nulls

def get_conversation_data(zip_code=None, start_date=None, end_date=None, sentiment=None, issue_category=None):
    """
    Retrieves raw conversation data (without embeddings) from BigQuery based on filters.
    Dates should be in 'YYYY-MM-DD' format.
    Queries the 'conversations' table.
    """
    query = f"""
    SELECT
        area,
        area_usps,
        content_tags,
        convo_id_url,
        data_source,
        date_post_published,
        date_post_received,
        district_usps,
        fb_tw,
        post_content,
        public_private,
        response_content,
        zip_category,
        zipcode,
        zipcode_num,
        unique_id
    FROM
        `{PROJECT_ID}.{DATASET_ID}.{CONVERSATIONS_TABLE}`
    WHERE
        1=1
    """

    # Track if any date filter was applied by arguments
    date_filter_applied = False
    if start_date:
        query += f" AND DATE(date_post_published) >= DATE('{start_date}')"
        date_filter_applied = True
    if end_date:
        query += f" AND DATE(date_post_published) <= DATE('{end_date}')"
        date_filter_applied = True

    if not date_filter_applied:
        query += f" AND DATE(date_post_published) >= DATE('{DEFAULT_BQ_START_DATE}') AND DATE(date_post_published) <= DATE('{DEFAULT_BQ_END_DATE}')"
        print(f"Applying default date filter from {DEFAULT_BQ_START_DATE} to {DEFAULT_BQ_END_DATE} for partitioned table.")


    if zip_code:
        query += f" AND zipcode_num = {zip_code}"
    if sentiment:
        query += f" AND LOWER(public_private) = LOWER('{sentiment}')"
    if issue_category:
        query += f" AND LOWER(content_tags) = LOWER('{issue_category}')"

    print(f"Executing BigQuery query for filtering...\n{query}")
    query_job = client.query(query)
    results = query_job.result()

    processed_results = []
    for row in results:
        row_dict = dict(row)
        
        # Explicitly cast unique_id to float for consistent JSON serialization
        unique_id_val = row_dict.get('unique_id')
        if unique_id_val is not None:
            try:
                unique_id_val = float(unique_id_val)
            except (ValueError, TypeError):
                unique_id_val = None # Or handle error as appropriate
        
        # Cast zipcode_num to int
        zipcode_num_val = row_dict.get('zipcode_num')
        if zipcode_num_val is not None:
            try:
                zipcode_num_val = int(zipcode_num_val)
            except (ValueError, TypeError):
                zipcode_num_val = None


        processed_results.append({
            'conversation_id': str(row_dict.get('convo_id_url')),
            'conversation_text': row_dict.get('post_content'),
            'sentiment': row_dict.get('public_private'),
            'issue_category': row_dict.get('content_tags'),
            'timestamp': get_and_format_date(row_dict, 'date_post_published'),
            'zip_code': zipcode_num_val, # Use the casted value
            'area': row_dict.get('area'),
            'area_usps': row_dict.get('area_usps'),
            'data_source': row_dict.get('data_source'),
            'date_post_received': get_and_format_date(row_dict, 'date_post_received'),
            'district_usps': row_dict.get('district_usps'),
            'fb_tw': row_dict.get('fb_tw'),
            'response_content': row_dict.get('response_content'),
            'zip_category': row_dict.get('zip_category'),
            'zipcode': row_dict.get('zipcode'),
            'unique_id': unique_id_val, # Use the casted value
        })

    print(f"Retrieved {len(processed_results)} rows from BigQuery after filtering.")
    return processed_results