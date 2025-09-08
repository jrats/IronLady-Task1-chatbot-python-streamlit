from openai import OpenAI 
from config import OPENAI_API_KEY, chat_model, similarity_threshold
from embeddings import embed_text
import faiss

client = OpenAI(api_key=OPENAI_API_KEY)

def retrieve_answer(query, index, questions, answers, k=1):
    if index is None or not questions:
        return None, None, 0.0

    query_vec = embed_text(query).reshape(1, -1)
    faiss.normalize_L2(query_vec)

    scores, idxs = index.search(query_vec, k)
    idx = int(idxs[0][0])  # convert to int in case of numpy type
    score = float(scores[0][0])

    # check if idx is valid
    if idx < 0 or idx >= len(questions):
        return None, None, score

    if score >= similarity_threshold:
        return questions[idx], answers[idx], score

    return None, None, score


def ask_gpt(user_message, retrieved_q=None, retrieved_a=None):
    sys_promt = "You are a helpful chatbot for Iron Lady's leadership programs."
    user_prompt = f"Question: {user_message}\n"
    if retrieved_q and retrieved_a:
        user_prompt += f'\nUse this FAQ info if relevant:\nQ: {retrieved_q}\nA:{retrieved_a}'
    completion = client.chat.completions.create(
        model = chat_model,
        messages=[
            {'role':'system', 'content': sys_promt},
            {'role':'user', 'content': user_prompt}
        ]
    )
    return completion.choices[0].message.content