# utils/global_vars.py

import os
from utils.token_logger import TokenCaptureLoguru

# Global token usage tracker (singleton instance)
token_tracker = TokenCaptureLoguru()

# Dataset name (modify here to switch datasets)
# Available options: [ProntoQA | ProofWriter | FOLIO | RepublicQA | ProverQA]
DATASET_NAME = "RepublicQA"

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data path
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", DATASET_NAME, "dev.json")

# Unified output file path (save all results)
RESULT_DIR = os.path.join(PROJECT_ROOT, "result", DATASET_NAME)
RESULT_JSON_PATH = os.path.join(RESULT_DIR, f"{DATASET_NAME}_result.jsonl")

# Prompt base directory (organized by dataset)
PROMPT_BASE_DIR = os.path.join(PROJECT_ROOT, "prompts", DATASET_NAME)

# Data buffer (assigned after main program loads)
dataset = None

# Currently processing sample ID (optional for debugging)
current_sample_id = None
