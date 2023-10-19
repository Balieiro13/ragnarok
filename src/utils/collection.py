import os

from dotenv import load_dotenv

import chromadb
from langchain.vectorstores import Chroma

from utils.embed import get_embedding_model

load_dotenv()


def get_chromadb_collection(collection_name: str) -> Chroma:
    embedding_function = get_embedding_model(os.getenv("EMBEDDING_MODEL_NAME")) 

    client = chromadb.HttpClient(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

    chroma_collection = Chroma(
        client=client, 
        collection_name=collection_name,
        embedding_function=embedding_function
    )
    return chroma_collection
