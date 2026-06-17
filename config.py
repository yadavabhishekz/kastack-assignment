import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


DATA_DIR       = ROOT_DIR / "data"
OUTPUTS_DIR    = ROOT_DIR / "outputs"

CSV_PATH            = DATA_DIR / "conversations.csv"
TOPIC_OUTPUT        = OUTPUTS_DIR / "topic_checkpoints.json"
CHECKPOINT_OUTPUT   = OUTPUTS_DIR / "checkpoint_100.json"
PERSONA_OUTPUT      = OUTPUTS_DIR / "persona.json"
CHROMA_DIR          = OUTPUTS_DIR / "chroma_db"


EMBEDDING_MODEL = "all-MiniLM-L6-v2"

GROQ_MODEL       = "llama-3.3-70b-versatile"
LLM_TEMPERATURE  = 0.3
LLM_MAX_TOKENS   = 500

# cosine similarity below this value = topic has changed
SIMILARITY_THRESHOLD = 0.5

CHECKPOINT_SIZE = 100   

TOP_K = 3   

PERSONA_BATCH_SIZE = 200   

TRIAL_MODE = True
TRIAL_MESSAGE_LIMIT = 5000