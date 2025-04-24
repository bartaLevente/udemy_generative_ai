from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="gemma2-9b-it",groq_api_key=groq_api_key)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate the following into {language}:"),
        ("human", "{text}"),
    ]
)

parser = StrOutputParser()

chain = prompt | llm | parser

app = FastAPI(  title="LangChain Server",
                version="0.0.1",
                description="LangChain Server for Groq LLM")


add_routes(
    app,
    chain,
    path="/translate"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost",port=8000)