from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os

def get_llm():
    return ChatMistralAI(model="mistral-small-latest",mistral_api_key=os.getenv("MISTRAL_API_KEY"),temperature=0.2)

def build_chain(system_promt:str):
    llm=get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) |
        ChatPromptTemplate.from_messages(
            [
                ("system", system_promt),
                ("human", "{text}"),
            ]
        ) | llm | StrOutputParser()
    )
    
def extract_action_items(transcript: str)->str:
    chain=build_chain(
        "You are a expert meeting analyst. From the meeting transcript,"
        "extract all action items for each provided:\n."
        "-Task description\n"
        "- Owner(Who is responsible for the task)\n"
        "- Deadline(if mentioned , else write 'Not satisfied')\n\n"
        "Format as numbered list. If not found say 'No action items found.'"
    )
    
    return chain.invoke(transcript)

def extract_key_decisions(transcript: str)->str:
    chain=build_chain(
        "you are a expert meeting analyst. From the meeting transcript,"
        "extract all key decisions and return them as a bullet list."
        "If there are no key decisions, return 'No key decisions found.'"
    )
    return chain.invoke(transcript)

def extract_questions(transcript: str)->str:
    chain=build_chain(
        "from the meeting transcript, extract all questions the unresolved questions or topics needing follow-up."
        "format as a numbered list. if none found say 'No open questions found.'"
    )
    return chain.invoke(transcript)