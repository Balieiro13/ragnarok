import os, json

from dotenv import load_dotenv

import chromadb

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


load_dotenv()

def get_chromadb_collection(collection_name: str) -> Chroma:
    embedding_function = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL_NAME"),
        model_kwargs=json.loads(os.getenv("EMBEDDING_MODEL_KWARGS")),
        encode_kwargs=json.loads(os.getenv("EMBEDDING_ENCODE_KWARGS")),
    )

    client = chromadb.HttpClient(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

    db = Chroma(
        client=client, 
        collection_name=collection_name,
        embedding_function=embedding_function
    )
    return db


def setup_chain(llm, template, **kwargs):
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt, **kwargs)
    return chain

