import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

db_uri = "USE_MYSQL"

api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Database Chatbot", page_icon=":parrot:")
st.title("Database Chatbot :parrot:")
st.write("This is a chatbot that can answer questions about a database.")

st.sidebar.header("Connent to Database")

mysql_host = st.sidebar.text_input("MySQL Host", "localhost")
mysql_port = st.sidebar.text_input("MySQL Port", "3306")
mysql_user = st.sidebar.text_input("MySQL User")
mysql_password = st.sidebar.text_input("MySQL Password", type="password")
mysql_db = st.sidebar.text_input("MySQL Database name")

@st.cache_resource(ttl="1h")
def configure(mysql_host, mysql_port, mysql_user, mysql_password, mysql_db):
    db_uri = f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}"
    engine = create_engine(db_uri)
    db = SQLDatabase(engine)

    return db


if not (mysql_host and mysql_port and mysql_user and mysql_password and mysql_db):
    st.sidebar.error("Please enter all the required fields.")
    st.stop()

db = configure(mysql_host, mysql_port, mysql_user, mysql_password, mysql_db)
st.sidebar.success("Connected to the database!")
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192",streaming=True)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
    

if "messages" not in st.session_state or st.sidebar.button("Clear chat"):
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

user_input = st.chat_input("Ask a question about the database:")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_input, callbacks=[streamlit_callback])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)