import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

faq_json = data_dir/"existing_faqs.json"
candidates_json = data_dir/"candidates_question.json"
index_file = data_dir/"faq_index.faiss"

embed_model = "text-embedding-3-small"     #less data
chat_model = "gpt-4o-mini"                 #less data
similarity_threshold = 0.82