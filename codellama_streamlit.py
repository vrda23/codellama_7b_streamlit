from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain import PromptTemplate
import streamlit as st
import time

# Custom prompt template
custom_prompt_template = """
You are an AI coding assistant. Solve the user's query given below:
Query: {query}
Return helpful code and related details:
"""

def set_custom_prompt():
    prompt = PromptTemplate(
        template = custom_prompt_template,
        input_variables = ["query"]
    )
    return prompt

def load_model():
    llm = CTransformers(
        model = "codellama-7b-instruct.ggmlv3.Q4_0.bin",
        model_type = "llama",
        max_new_tokens = 1096,
        temperature = 0.2,
        repetition_penalty = 1.13
    )
    return llm

def chain_pipeline():
    llm = load_model()
    qa_prompt = set_custom_prompt()
    qa_chain = LLMChain(
        prompt = qa_prompt,
        llm = llm
    )
    return qa_chain

llm_chain = chain_pipeline()

# Streamlit UI
st.title("Codellama-7b Demo")
user_input = st.text_area("Enter your query here:")
submit_button = st.button("Submit")

# Chat history
chat_history = []

if submit_button:
    llm_response = llm_chain.run({"query": user_input})
    # Append to chat history
    chat_history.append(("User:", user_input))
    chat_history.append(("Bot:", llm_response))
    for user, text in chat_history:
        st.write(f"{user} {text}")
