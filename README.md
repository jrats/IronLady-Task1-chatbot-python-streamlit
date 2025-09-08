Iron Lady Chatbot

A Streamlit-based RAG chatbot that answers FAQs related to Iron Lady's leadership programs, with functionality to add and promote new FAQs.

------------------------------------------------------------------

🛠️ Tools / Tech Stack

Python 3.10+

Streamlit – Frontend / UI

OpenAI API – GPT-4o-mini for answering questions

FAISS – Vector store for similarity search of FAQs

NumPy – Handling embeddings

python-dotenv – Load API keys

JSON – Store FAQs and candidate questions

------------------------------------------------------------------

⚡ Features Implemented

FAQ Chatbot: Ask questions about Iron Lady programs.

RAG (Retrieval-Augmented Generation): Retrieves closest FAQ via FAISS embeddings and uses GPT to generate answers.

Add New FAQs: Admins can add new FAQs via sidebar.

Candidate Questions: Tracks user questions not in FAQ.

Promote Candidates: Admin can promote frequently asked questions to FAQs and provide an answer.

Automatic Answer Guidance: GPT references closest FAQ if available; otherwise, suggests visiting the Iron Lady website.

Persistent Storage: FAQs, candidate questions, and FAISS index are stored in data/.

------------------------------------------------------------------

📦 Project Structure

ironlady_chatbot/
│
├── chatbot_ui.py         # Streamlit UI
├── config.py             # Config & constants
├── embeddings.py         # Embeddings + FAISS
├── storage.py            # Load/save JSON data
├── chatbot.py            # GPT calls + logic
├── data/
│   ├── faq_data.json
│   ├── candidate_questions.json
│   └── faq_index.faiss
├── requirements.txt
└── README.md

------------------------------------------------------------------

🚀 How to Run

Clone the repo:

git clone https://github.com/jrats/IronLady-Task1-chatbot-python-streamlit.git

cd ironlady_chatbot


Install dependencies:

pip install -r requirements.txt


Create a .env file with your OpenAI API key:


OPENAI_API_KEY=your_openai_api_key


Run the Streamlit app:

streamlit run chatbot_ui.py


Open the browser to interact with the chatbot.

------------------------------------------------------------------

🔑 Notes

The chatbot uses FAISS to find similar FAQs and GPT-4o-mini to generate answers.

Admin sidebar allows adding/promoting FAQs dynamically.

Candidate questions are saved and can be reviewed/promoted later.

------------------------------------------------------------------
