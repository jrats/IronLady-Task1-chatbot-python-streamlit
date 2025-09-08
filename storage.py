import json
from config import faq_json, candidates_json


def save_faqs(questions, answers):
    faq_json.write_text(json.dumps({'questions': questions, 'answers': answers}, indent=2))

def load_faqs():
    if faq_json.exists():
        data = json.loads(faq_json.read_text())
        return data['questions'], data['answers']
    return [], []

def save_candidates(candidates):
    candidates_json.write_text(json.dumps(candidates, indent=2))

def load_candidates():
    if candidates_json.exists():
        return json.loads(candidates_json.read_text())