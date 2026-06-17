import os
import sys
import json
import streamlit as st
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))


from config import GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, PERSONA_OUTPUT


from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


from retrieval.rag import retrieve, build_context_string, index_topic_summaries, index_message_checkpoints




llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
    groq_api_key=GROQ_API_KEY,
)




PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["persona_text", "context_text", "question"],
    template="""You are a helpful assistant that answers questions about a person based on their conversation history.

PERSONA DATA:
{persona_text}

RELEVANT CONVERSATION CONTEXT:
{context_text}

USER QUESTION:
{question}

Instructions:
- Answer based ONLY on the persona data and conversation context provided above
- Be specific and cite evidence when possible
- If the answer is not in the data, say "I don't have enough information about that"
- Keep the answer clear and concise

Answer:""",
)





chain = PROMPT_TEMPLATE | llm



@st.cache_resource
def load_indexes():
    
    index_topic_summaries()
    index_message_checkpoints()
    return True


@st.cache_data
def load_persona():
    
    if not PERSONA_OUTPUT.exists():
        return {}
    with open(PERSONA_OUTPUT, "r", encoding="utf-8") as f:
        return json.load(f)


def answer_question(question: str, persona: dict) -> str:
    
    
    retrieval_results = retrieve(question, top_k=3)
    context_text = build_context_string(retrieval_results)

    
    persona_text = json.dumps(persona, indent=2) if persona else "No persona data available."

    
    
    response = chain.invoke({
        "persona_text": persona_text,
        "context_text": context_text,
        "question": question,
    })

    
    return response.content.strip()



def main():
    st.set_page_config(
        page_title="Conversation Analyzer Chatbot",
        page_icon="💬",
        layout="centered"
    )

    st.title("💬 Conversation Analyzer Chatbot")
    st.markdown("""
    Ask me anything about the user based on their conversation history.
    
    **Try asking:**
    - What kind of person is this user?
    - What are their habits?
    - What are their interests?
    - How do they communicate?
    - What topics do they talk about most?
    """)

    
    with st.spinner("Loading data and indexes..."):
        load_indexes()
        persona = load_persona()

    
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    
    if question := st.chat_input("Ask a question about the user..."):
        
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_question(question, persona)
            st.markdown(answer)

        
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
