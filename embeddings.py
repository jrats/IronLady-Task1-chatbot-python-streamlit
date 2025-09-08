import faiss
from openai import OpenAI
import numpy as np
from config import OPENAI_API_KEY, embed_model, index_file

client = OpenAI(api_key=OPENAI_API_KEY)

def embed_text(text):
    emb = client.embeddings.create(model = embed_model, input = text)
    return np.array(emb.data[0].embedding, dtype='float32')

def build_index(questions):
    if not questions:
        return None
    embeddings = [embed_text(q) for q in questions]
    embeddings = np.vstack(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_file)) #saving instead of rebuilding
    return index
    
def load_index():
    if index_file.exists():
        return faiss.read_index(str(index_file)) #faiss expects str path
    return None