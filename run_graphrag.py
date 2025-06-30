import os
from dotenv import load_dotenv
import yaml
import shutil
import subprocess
import sys

# Define the path to your configuration file (should be settings.yaml)
CONFIG_PATH = "settings.yaml"
ROOT_DIR = "./ragtest_project" # A dedicated root directory for GraphRAG output and input

# --- NEW: Get the path to the graphrag executable in the venv ---
# On Windows, executables are typically in Scripts
# On Linux/macOS, they are typically in bin
# We'll use os.path.join to be cross-platform safe
if sys.platform == "win32":
    GRAPHRAG_CLI_PATH = os.path.join(sys.prefix, "Scripts", "graphrag.exe")
else:
    GRAPHRAG_CLI_PATH = os.path.join(sys.prefix, "bin", "graphrag")

# You might want to add a check here to ensure the executable exists
if not os.path.exists(GRAPHRAG_CLI_PATH):
    raise FileNotFoundError(f"GraphRAG CLI executable not found at: {GRAPHRAG_CLI_PATH}. "
                            "Ensure GraphRAG is correctly installed in your virtual environment.")


def setup_graphrag_workspace():
    """
    Sets up the GraphRAG workspace by running `graphrag init`.
    """
    print(f"--- Setting up GraphRAG Workspace in {ROOT_DIR} ---")
    os.makedirs(ROOT_DIR, exist_ok=True) # Ensure the root directory exists

    # Clean up previous outputs/configs if they exist for a fresh run
    # Ensure these are cleaned from within the ROOT_DIR
    if os.path.exists(os.path.join(ROOT_DIR, 'output')):
        print(f"Cleaning up existing '{ROOT_DIR}/output' directory...")
        shutil.rmtree(os.path.join(ROOT_DIR, 'output'))
    if os.path.exists(os.path.join(ROOT_DIR, CONFIG_PATH)):
        print(f"Cleaning up existing '{ROOT_DIR}/{CONFIG_PATH}'...")
        os.remove(os.path.join(ROOT_DIR, CONFIG_PATH))
    # It's safest to clean the .env inside ROOT_DIR as well, as graphrag init might overwrite it
    if os.path.exists(os.path.join(ROOT_DIR, '.env')):
        print(f"Cleaning up existing '{ROOT_DIR}/.env'...")
        os.remove(os.path.join(ROOT_DIR, '.env'))
    # Also clean up the prompts directory, as graphrag init generates it
    if os.path.exists(os.path.join(ROOT_DIR, 'prompts')):
        print(f"Cleaning up existing '{ROOT_DIR}/prompts' directory...")
        shutil.rmtree(os.path.join(ROOT_DIR, 'prompts'))


    # Run `graphrag init` command as a subprocess
    init_command = [GRAPHRAG_CLI_PATH, "init", "--root", ROOT_DIR]
    print(f"Running command: {' '.join(init_command)}")
    try:
        subprocess.run(init_command, check=True, capture_output=True, text=True)
        print("GraphRAG workspace initialized successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error initializing GraphRAG workspace: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise

    # 2. Update .env and settings.yaml (critical for Groq integration)
    # ***CORRECTED LINE: env_file_path should be inside ROOT_DIR***
    env_file_path = os.path.join(ROOT_DIR, ".env")
    settings_file_path = os.path.join(ROOT_DIR, CONFIG_PATH)

    # Update .env with GROQ_API_KEY
    groq_api_key = os.getenv("GROQ_API_KEY")
    print("this is groq_api_key", groq_api_key) # This print is helpful for debugging
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set. Please set it in your system or .env file.")

    # Load the .env file that `graphrag init` just created, modify it, and rewrite it.
    # This is safer than just appending, in case `graphrag init` puts a placeholder that needs replacing.
    env_lines = []
    with open(env_file_path, 'r') as f:
        env_lines = f.readlines()

    # Find and replace the GRAPHRAG_API_KEY line, or append if not found
    found_key = False
    for i, line in enumerate(env_lines):
        if line.strip().startswith("GRAPHRAG_API_KEY="):
            env_lines[i] = f"GRAPHRAG_API_KEY={groq_api_key}\n"
            found_key = True
            break
    if not found_key:
        env_lines.append(f"\nGRAPHRAG_API_KEY={groq_api_key}\n") # Append if it wasn't there

    with open(env_file_path, 'w') as f:
        f.writelines(env_lines)
    print(f"Updated '{env_file_path}' with GRAPHRAG_API_KEY.")

    # Update settings.yaml for Groq (OpenAI-compatible API)
    with open(settings_file_path, 'r') as f:
        settings_config = yaml.safe_load(f)

    # Configure LLM section for Groq (using OpenAI-compatible settings)
    settings_config['llm'] = {
        'api_key': '${GRAPHRAG_API_KEY}', # Reference environment variable from the .env file within ROOT_DIR
        'type': 'openai_chat',
        'api_base': "https://api.groq.com/openai/v1",
        'model': 'llama3-8b-8192',
        'model_supports_json': True,
        'max_tokens': 8192,
        'temperature': 0.0,
        'concurrent_requests': 10
    }

    settings_config['embeddings'] = {
        'llm': {
            'api_key': '${GRAPHRAG_API_KEY}',
            'type': 'openai_embedding',
            'api_base': "https://api.groq.com/openai/v1",
            'model': 'nomic-embed-text', # **Review this again for Groq compatibility**
            'token_limit': 8192
        },
        'concurrent_requests': 10
    }

    # Write the updated settings back to the file
    with open(settings_file_path, 'w') as f:
        yaml.safe_dump(settings_config, f, default_flow_style=False)
    print(f"Updated '{settings_file_path}' for Groq LLM and Embeddings configuration.")

    # 3. Prepare Input Data (as per documentation)
    input_dir = os.path.join(ROOT_DIR, "input")
    os.makedirs(input_dir, exist_ok=True)
    sample_text_file = os.path.join(input_dir, "sample_document.txt")
    if not os.path.exists(sample_text_file):
        print(f"Creating sample input document: {sample_text_file}")
        with open(sample_text_file, 'w', encoding='utf-8') as f:
            f.write("This is a sample document about GraphRAG and its capabilities. GraphRAG helps in extracting entities and relationships to build a knowledge graph. It uses large language models for various tasks like entity extraction and summarization. This document also discusses the importance of setting up the environment variables correctly and configuring the settings.yaml file for optimal performance. Remember to install graphrag with the necessary extras.")
    print(f"Input data ready in '{input_dir}'.")


def run_graphrag_indexing_pipeline():
    """
    Runs the GraphRAG indexing pipeline using the CLI command.
    """
    print("--- Running GraphRAG Indexing Pipeline ---")
    index_command = [GRAPHRAG_CLI_PATH, "index", "--root", ROOT_DIR]
    print(f"Running command: {' '.join(index_command)}")
    try:
        subprocess.run(index_command, check=True, capture_output=True, text=True)
        print("GraphRAG indexing pipeline completed successfully.")
        output_dir = os.path.join(ROOT_DIR, "output")
        print(f"Outputs are located in: {output_dir}")
        print("You can now inspect the generated parquet files for entities, relationships, and reports.")
    except subprocess.CalledProcessError as e:
        print(f"Error running GraphRAG indexing pipeline: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise


if __name__ == "__main__":
    try:
        # First, set up the workspace and configuration files
        setup_graphrag_workspace()

        # Then, run the indexing pipeline
        run_graphrag_indexing_pipeline()

        print("\n--- GraphRAG Pipeline Execution Finished ---")
        print(f"To query the graph, use the CLI: {GRAPHRAG_CLI_PATH} query --root {ROOT_DIR} --method global --query 'What are the main topics?'")

    except Exception as e:
        print(f"An unrecoverable error occurred: {e}")
        import traceback
        traceback.print_exc()
        
        


