import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the question based on the context provided. If the answer is not in the context, say 'I don't know'.\n\nContext: {context}\n\nQuestion: {input}\n\nAnswer:"
)

def create_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        st.session_state.loader = PyPDFLoader("attention.pdf")
        st.session_state.documents = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents)
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )

user_prompt = st.text_input("Enter your question about attention paper:")

if st.button("Create vectorstore"):
    create_embeddings()
    st.write("Database ready.")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    st.write("Response time:", time.process_time() - start)
    st.write("Response:", response["answer"])

    with st.expander("Source documents"):
        for i,doc in enumerate(response["context"]):
            st.write(f"Document {i+1}:")
            st.write(doc.page_content)
            st.write("-----------------------------------------------------------------------------------")