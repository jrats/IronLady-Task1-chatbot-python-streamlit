import streamlit as st
from collections import Counter

from storage import save_faqs, save_candidates, load_faqs, load_candidates
from embeddings import build_index, load_index
from chatbot import retrieve_answer, ask_gpt

st.set_page_config(page_title='Iron Lady Chatbot', layout='wide')

st.title('Iron LadyChatbot')

#initialise session state with necessary variables
if 'questions' not in st.session_state:
    st.session_state['questions'], st.session_state['answers'] = load_faqs()
if 'index' not in st.session_state:
    st.session_state['index'] = load_index() or build_index(st.session_state['questions'])
if 'candidates' not in st.session_state or not st.session_state['candidates']:
    st.session_state['candidates'] = load_candidates() or []
if 'promoting_q' not in st.session_state:
    st.session_state['promoting_q'] = None

#Chat interface
if prompt := st.chat_input("Ask me about Iron Lady's programs..."):
    st.session_state['messages'] = st.session_state.get('messages',[])
    st.session_state['messages'].append({'role':"user", "content": prompt})

    retrieved_q, retrieved_a, score = retrieve_answer(prompt, st.session_state['index'], st.session_state['questions'], st.session_state['answers'], k=1)

    if retrieved_q:
        response = ask_gpt(prompt, retrieved_q, retrieved_a)
    else:
        response = ask_gpt(prompt) + '\n For more details, please visit: https://iamironlady.com/'
        st.session_state['candidates'].append(prompt)
        save_candidates(st.session_state['candidates'])
    st.session_state['messages'].append({'role':'assistant', 'content': response})

#give chat history
for msg in st.session_state.get('messages',[]):
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

#sidebar with admin tools
#can protect this with authentication
st.sidebar.header('Admin Tools')

#to add faqs
with st.sidebar.expander('Add New FAQs'):
    new_q = st.text_input('Question')
    new_a = st.text_area('Answer')
    if st.button('Add FAQ'):
        if new_q and new_a:
            st.session_state['questions'].append(new_q)
            st.session_state['answers'].append(new_a)
            save_faqs(st.session_state['questions'], st.session_state['answers'])
            st.session_state['index'] = build_index(st.session_state['questions'])
            st.success("FAQ added successfully!")

#questions asked by users multiple times that are not part of the FAQs
with st.sidebar.expander('Candidate FAQs'):
    counter = Counter(st.session_state['candidates'])
    for q, count in counter.most_common():
        st.write(f"-**{q}** (asked {count} times)")
        #have the option of promoting questions asked multiple times to FAQs and type an appropriate answer
        if st.button(f"Promote '{q}' to FAQ", key=f"promote_{q}"):
            st.session_state['promoting_q']=q

if st.session_state['promoting_q']:
    q = st.session_state['promoting_q']
    new_a = st.text_area(f"Answer for '{q}'", key=f"ans_{q}")
    if st.button(f"Save FAQ for '{q}'", key=f"save_{q}"):
        if new_a.strip():
            #add to faq
            st.session_state['questions'].append(q)
            st.session_state['answers'].append(new_a.strip())

            #save faqs and rebuild index
            save_faqs(st.session_state['questions'], st.session_state['answers'])
            st.session_state['index'] = build_index(st.session_state['questions'])

            #remove from candidates
            st.session_state['candidates'] = [c for c in st.session_state['candidates'] if c!=q ]
            save_candidates(st.session_state['candidates'])


            #reset promotion state
            del st.session_state['promoting_q']

            st.success(f"Promoted {q} to FAQ and answer added!")

        else:
            st.warning('Please enter an answer before saving.')