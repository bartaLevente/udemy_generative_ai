import streamlit as st
import validators
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document 
from youtube_transcript_api import YouTubeTranscriptApi
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Summarize content of YT video or Webpage by URL", page_icon=":book:")
st.title("Summarize content of YT video or Webpage by URL :book:")

api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192")

prompt_template = """You are a helpful assistant that summarizes the content of a document.
You will be provided with a document and you will summarize it in a concise manner.
Please provide a summary of the document in a few sentences.
Document: {text}
Summary:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

url = st.text_input("Enter the URL of the YT video or Webpage to summarize:")

button = st.button("Summarize!")

if button:
    if not url:
        st.error("Please enter a URL.")
    elif not validators.url(url):
        st.error("Please enter a valid URL.")
    try:
        with st.spinner("Summarizing..."):
            if "youtube.com" in url:
                video_id = url.split("v=")[-1]
                transcript = YouTubeTranscriptApi.get_transcript(video_id=video_id)
                text = " ".join([entry['text'] for entry in transcript])
                data = [Document(page_content=text)]
            else:
                loader = UnstructuredURLLoader(urls=[url],ssl_verify=False, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                data = loader.load()
            
            chain = load_summarize_chain(llm, chain_type="stuff", verbose=True, prompt=prompt)
            summary = chain.run(data)

            st.success(summary)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please check the URL and try again.")
