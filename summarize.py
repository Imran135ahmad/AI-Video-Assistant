from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter # ignore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os

def get_llm():
    return ChatMistralAI(model="mistral-small-latest",mistral_api_key=os.getenv("MISTRAL_API_KEY"),temperature=0.3)

def split_transcript(transcript: str)->list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return text_splitter.split_text(transcript)

def summarize(transcript: str)-> str:
    llm = get_llm()
    map_promt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that summarizes transcripts of audio recordings."),
            ("human","{text}")
        ]
    )
    map_chain=map_promt | llm | StrOutputParser()
    
    chunks = split_transcript(transcript)
    
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]
    
    combined = "\n\n".join(chunk_summaries)
    
    combine_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert meeting summarizer. You will be given partial meeting summaries. "
            "Combine them into one final professional summary in bullet points. "
            "Remove duplicate information and make the summary concise and clear."
        ),
        ("human", "{text}")
    ]
)
    
    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | combine_prompt | llm | StrOutputParser()
        )
    return combined_chain.invoke(combined)

def generate_title(transcript: str)->str:
    llm=get_llm()
    
    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) |
        ChatPromptTemplate.from_messages([
            ("system",
             "Based on the meeting transcript, generate a short professional meeting title"
             "(max 8 words). Only return the title nothing else."
            ),
            ("human", "{text}")
        ]
        ) | llm | StrOutputParser()                                 
)
    
    return title_chain.invoke(transcript[:2000])