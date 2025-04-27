import streamlit as st
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, LLMMathChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.callbacks import StreamlitCallbackHandler
load_dotenv()
from langchain.agents.agent_types import AgentType

api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it", streaming=True)

prompt_template = PromptTemplate(
    input_variables=["question"],
    template="You are a math expert. Answer the question: {question}"
)

math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Solve the following math problem and return ONLY the final answer, NO code, NO explanations, NO numexpr, only the final number in plain text.",
)

wiki_wrapper = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia",
    func=wiki_wrapper.run,
    description="A tool for searching information on the web.",
)

reason_chain = LLMChain(llm=llm, prompt=prompt_template)
reason_tool = Tool(
    name="Reasoning",
    func=reason_chain.run,
    description="Useful for when you need to answer logical based questions.",
)

tools = [calculator, wiki_tool, reason_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)


st.set_page_config(page_title="Math Chatbot", page_icon=":parrot:")
st.title("Math Chatbot :parrot:")
st.write("This is a chatbot that can answer questions about math.")

user_input = st.text_input("Enter your math question here:", key="question")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

button = st.button("Submit")
if button:
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
    else:
        st.error("Please enter a question.")