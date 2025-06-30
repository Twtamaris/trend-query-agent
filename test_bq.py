from google.cloud import bigquery
import pandas as pd
from datetime import datetime, timedelta
import os # Import os to access environment variables

# Import LangChain components for GROQ and prompting
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import re
load_dotenv()

# Initialize BigQuery client
client = bigquery.Client()


try:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    llm = ChatGroq(temperature=0.0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")
    # You can choose other models like "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"
    # Llama3-8b-8192 is a good balance of speed and capability for summarization.
except ValueError as e:
    print(f"Error initializing ChatGroq: {e}")
    print("Please set the GROQ_API_KEY environment variable.")
    exit() # Exit if API key is not found

# --- BigQuery Data Retrieval Function (Mostly unchanged, with project_id fix) ---
def get_conversation_data(zip_code=None, start_date=None, end_date=None, sentiment=None, issue_category=None):
    """
    Retrieves conversation data from BigQuery based on filters.
    Dates should be in 'YYYY-MM-DD' format.
    """
    # Replace with your actual project ID
    project_id = "chatbot-project-464108"

    query = f"""
    SELECT
        conversation_text,
        sentiment,
        issue_category,
        timestamp,
        zip_code
    FROM
        `{project_id}.customer_service_data.conversations`
    WHERE
        1=1
    """

    if zip_code:
        query += f" AND zip_code = '{zip_code}'"
    if start_date:
        query += f" AND DATE(timestamp) >= DATE('{start_date}')"
    if end_date:
        query += f" AND DATE(timestamp) <= DATE('{end_date}')"
    if sentiment:
        # Use LIKE for case-insensitivity or ensure your data's sentiment is standardized
        # For exact match, use = '{sentiment}'
        query += f" AND LOWER(sentiment) = LOWER('{sentiment}')" # Convert to lowercase for matching
    if issue_category:
        query += f" AND LOWER(issue_category) = LOWER('{issue_category}')" # Convert to lowercase for matching

    print(f"Executing BigQuery query:\n{query}") # For debugging
    query_job = client.query(query)
    results = query_job.result()
    return [dict(row) for row in results]

# --- LLM-Powered Summarization Function ---
def summarize_conversations_with_llm(conversations):
    """
    Summarizes conversations, identifies key issues, and overall sentiment using an LLM.
    """
    if not conversations:
        return {
            "summary": "No conversations found for the given criteria.",
            "issues": [],
            "overall_sentiment": "N/A"
        }

    # Combine conversation texts for the LLM
    all_texts = [conv['conversation_text'] for conv in conversations if conv['conversation_text']]
    if not all_texts:
        return {
            "summary": "No conversation text available.",
            "issues": [],
            "overall_sentiment": "N/A"
        }

    # Limit the number of conversations to send to the LLM to manage context window and cost
    # For very many conversations, consider summarizing in batches or extracting key info from each
    # and then summarizing the summaries.
    conversations_for_llm = "\n---\n".join(all_texts[:5]) # Take up to first 5 conversations for brevity

    # Define the prompt template for the LLM
    # This prompt tells the LLM what to do and what format to return the answer.
    # We ask for a JSON output for easier parsing.
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert customer service assistant. Your task is to analyze customer conversations.
                Summarize the main points, identify distinct issues mentioned, and determine the overall sentiment.
                Provide the output in a JSON format with the following keys:
                - 'summary': A concise summary of all conversations.
                - 'issues': A list of unique, distinct issues mentioned by customers.
                - 'overall_sentiment': The overall sentiment (Positive, Negative, Neutral, or Mixed) across all conversations.
                Example JSON output:
                ```json
                {{
                    "summary": "Customer service issues included internet slowness and billing disputes.",
                    "issues": ["Internet speed", "Billing error"],
                    "overall_sentiment": "Negative"
                }}
                ```
                """
            ),
            ("human", "Analyze the following customer service conversations:\n\n{conversations}"),
        ]
    )

    # Create an LLM chain
    llm_chain = prompt_template | llm | StrOutputParser() # StrOutputParser converts the LLM's Message to a string
    

    try:
        llm_response = llm_chain.invoke({"conversations": conversations_for_llm})
        print(f"\nLLM Raw Response:\n{llm_response}") # For debugging LLM output

        # --- MODIFIED PART ---
        # Try to extract JSON from the response using regex
        # This regex looks for a block starting with '{' and ending with '}'
        # and captures everything in between. It's non-greedy.
        json_match = re.search(r'\{(.*?)\}', llm_response, re.DOTALL)

        if json_match:
            # If a JSON-like block is found, extract it
            json_string = "{" + json_match.group(1) + "}"
        else:
            # Fallback: if no JSON block found, try to strip potential markdown
            # This is less reliable if the LLM puts a lot of text before/after.
            if llm_response.startswith("```json") and llm_response.endswith("```"):
                json_string = llm_response[7:-3].strip()
            else:
                json_string = llm_response.strip()

        # Attempt to parse the JSON string
        parsed_response = json.loads(json_string)

        return {
            "summary": parsed_response.get("summary", "No summary provided by LLM."),
            "issues": parsed_response.get("issues", []),
            "overall_sentiment": parsed_response.get("overall_sentiment", "N/A")
        }

    except Exception as e:
        print(f"Error during LLM call or parsing: {e}")
        # Fallback to basic summary if LLM fails
        fallback_summary_text = f"LLM summarization failed. Found {len(conversations)} conversations.\n"
        fallback_issues = {}
        fallback_sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
        for conv in conversations:
            if conv.get('issue_category'):
                fallback_issues[conv['issue_category']] = fallback_issues.get(conv['issue_category'], 0) + 1
            if conv.get('sentiment'):
                fallback_sentiments[conv['sentiment'].lower()] = fallback_sentiments.get(conv['sentiment'].lower(), 0) + 1

        return {
            "summary": fallback_summary_text,
            "issues": list(fallback_issues.keys()),
            "overall_sentiment": next(iter(fallback_sentiments.keys())) if fallback_sentiments else "N/A"
        }


# --- Example Usage (simulating chatbot interaction with LLM) ---
if __name__ == "__main__":
    import json # Import json for parsing LLM output

    # Example 1: Summarize conversations for a specific zip code in the last 7 days
    today = datetime.now()
    seven_days_ago = today - timedelta(days=7)
    start_date_str = seven_days_ago.strftime('%Y-%m-%d')
    end_date_str = today.strftime('%Y-%m-%d')
    # Use a zip code that actually exists in your sample data for testing
    my_zip_code = '90210' # Example from your data
    my_start_date = '2025-06-20' # Dates from your sample data to ensure results
    my_end_date = '2025-06-22'

    print(f"\n--- Querying for zip code {my_zip_code} in last few days ---")
    relevant_convs = get_conversation_data(
        zip_code=my_zip_code,
        start_date=my_start_date, 
        end_date=my_end_date      
    )
    
    print("OUTPUT OF GET CONVERSATION DATA", relevant_convs)

    llm_summary_output = summarize_conversations_with_llm(relevant_convs)
    print("\n--- LLM-Generated Summary ---")
    print(f"Summary: {llm_summary_output['summary']}")
    print(f"Issues: {', '.join(llm_summary_output['issues'])}")
    print(f"Overall Sentiment: {llm_summary_output['overall_sentiment']}")

    # Example 2: What are the specific issues people are facing with negative sentiment?
    print(f"\n--- Querying for negative sentiment issues in zip code {my_zip_code} ---")
    negative_issues_convs = get_conversation_data(
        zip_code=my_zip_code,
        sentiment='Negative', # Match case in your sample data or use LOWER() in query
        start_date=my_start_date,
        end_date=my_end_date
    )

    llm_negative_summary_output = summarize_conversations_with_llm(negative_issues_convs)
    print("\n--- LLM-Generated Summary for Negative Issues ---")
    print(f"Summary: {llm_negative_summary_output['summary']}")
    print(f"Issues: {', '.join(llm_negative_summary_output['issues'])}")
    print(f"Overall Sentiment: {llm_negative_summary_output['overall_sentiment']}")






















# from google.cloud import bigquery
# import pandas as pd
# from datetime import datetime, timedelta
# import os
# from dotenv import load_dotenv

# load_dotenv()


# # Initialize BigQuery client
# client = bigquery.Client()

# def get_conversation_data(zip_code=None, start_date=None, end_date=None, sentiment=None, issue_category=None):
#     """
#     Retrieves conversation data from BigQuery based on filters.
#     Dates should be in 'YYYY-MM-DD' format.
#     """
#     query = f"""
#     SELECT
#         conversation_text,
#         sentiment,
#         issue_category,
#         timestamp,
#         zip_code
#     FROM
#         `chatbot-project-464108.customer_service_data.conversations`
#     WHERE
#         1=1
#     """

#     if zip_code:
#         query += f" AND zip_code = '{zip_code}'"
#     if start_date:
#         query += f" AND DATE(timestamp) >= DATE('{start_date}')"
#     if end_date:
#         query += f" AND DATE(timestamp) <= DATE('{end_date}')"
#     if sentiment:
#         query += f" AND sentiment = '{sentiment}'"
#     if issue_category:
#         query += f" AND issue_category = '{issue_category}'"

#     print(f"Executing BigQuery query:\n{query}") # For debugging
#     query_job = client.query(query)
#     results = query_job.result()
#     return [dict(row) for row in results]

# def summarize_conversations(conversations):
#     """
#     A very basic summarizer. In a real system, you'd use an LLM here.
#     """
#     if not conversations:
#         return "No conversations found for the given criteria."

#     # For now, let's just return a count and some sample texts
#     summary_text = f"Found {len(conversations)} conversations.\n"
#     issues = {}
#     sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}

#     for conv in conversations:
#         summary_text += f"- Conversation on {conv['timestamp'].strftime('%Y-%m-%d %H:%M')}: {conv['conversation_text'][:100]}...\n" # show first 100 chars
#         if conv['issue_category']:
#             issues[conv['issue_category']] = issues.get(conv['issue_category'], 0) + 1
#         if conv['sentiment']:
#             sentiments[conv['sentiment']] = sentiments.get(conv['sentiment'], 0) + 1

#     if issues:
#         summary_text += "\nCommon issues:\n" + "\n".join([f"  - {issue}: {count}" for issue, count in issues.items()])
#     if sentiments:
#         summary_text += "\nOverall sentiments:\n" + "\n".join([f"  - {s}: {count}" for s, count in sentiments.items()])

#     return summary_text


# # --- Example Usage (simulating chatbot interaction) ---
# if __name__ == "__main__":
#     # Example 1: Summarize conversations for a specific zip code in the last 7 days
#     today = datetime.now()
#     seven_days_ago = today - timedelta(days=7)
#     start_date_str = seven_days_ago.strftime('%Y-%m-%d')
#     end_date_str = today.strftime('%Y-%m-%d')
#     my_zip_code = '90210' # Replace with a zip code from your data

#     print(f"\n--- Querying for zip code {my_zip_code} in last 7 days ---")
#     relevant_convs = get_conversation_data(
#         zip_code=my_zip_code,
#         start_date=start_date_str,
#         end_date=end_date_str
#     )
#     print(summarize_conversations(relevant_convs))

#     # Example 2: What are the specific issues people are facing in a particular geography and time period?
#     # This assumes your data has an 'issue_category' column.
#     print(f"\n--- Querying for issues in zip code {my_zip_code} for negative sentiment ---")
#     negative_issues_convs = get_conversation_data(
#         zip_code=my_zip_code,
#         sentiment='Negative', # Assuming your sentiment column has 'negative'
#         start_date=start_date_str,
#         end_date=end_date_str
#     )
#     print(summarize_conversations(negative_issues_convs))