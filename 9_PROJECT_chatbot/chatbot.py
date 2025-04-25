import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from operator import itemgetter
import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OpenAI"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please answer the questions"),
        ("user", "Question: {question}"),
    ]
)

output_parser = StrOutputParser()

def generate_response(question, api_key, llm, temprature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm, temperature=temprature, max_tokens=max_tokens)
    chain = prompt | llm | output_parser

    answer = chain.invoke({"question": question})
    return answer


st.title("Q&A Chatbot with OpenAI")
st.write("This is a simple Q&A chatbot using OpenAI's language model.")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

llm= st.sidebar.selectbox(
    "Select the LLM model",
    ["gpt-3.5-turbo", "gpt-4o-mini"]
)

temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

st.write("## Ask a question")
user_input = st.text_input("You:")

if user_input:
    try:
        response = generate_response(user_input, api_key, llm, temp, max_tokens)
        st.write(response)
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.write("Please enter a question to get a response.")
