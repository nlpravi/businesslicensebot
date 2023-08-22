import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.document_loaders import *
from langchain.chains.summarize import load_summarize_chain
import tempfile
from langchain.docstore.document import Document

st.title('Business Licenses Helper Bot')

def citationAnswerGenerator(business_jurisdiction):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0
    )
    system_template = """You are an assistant designed to provide answers to user's questions with a citation or reference. The question is related to business jurisdiction."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """The jurisdiction is {business_jurisdiction}. Please provide an answer to the question with a citation or reference."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(business_jurisdiction=business_jurisdiction)
    return result # returns string   

def display_answer(answer):
    if answer != "":
        st.markdown(f"**Answer:** {answer}")
    else:
        st.markdown("Please enter your business type and jurisdiction to get an answer.")

def unknownResponseGenerator():
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0
    )
    system_template = """You are an AI assistant. If you don't know the answer to a question, you should respond politely and honestly."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Please generate a response indicating that you don't know the answer."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run({})
    return result # returns string   

def display_unknown_response(unknown_response):
    if unknown_response != "":
        st.markdown(f"**Response:** {unknown_response}")
    else:
        st.markdown("The system is unable to generate a response at this time.")

#Get the business type and jurisdiction from the user
business_jurisdiction = st.text_input('Please enter your business type and jurisdiction')

#Create a button to trigger the functionality of the app
if st.button('Generate Answer'):
    if business_jurisdiction:
        answer = citationAnswerGenerator(business_jurisdiction)
        display_answer(answer)
        unknown_response = unknownResponseGenerator()
        display_unknown_response(unknown_response)
    else:
        st.markdown("Please enter your business type and jurisdiction to get an answer.")
