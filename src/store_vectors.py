import os
import uuid
import argparse
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

def main(dir_path, collection, reset=False):
    loader = DirectoryLoader(
        dir_path,
        glob="*.pdf", 
        show_progress=True,
        use_multithreading=True,
        loader_cls=PyPDFLoader
    )

    docs = loader.load()

    # Spliting text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=20)
    docs_splitted = text_splitter.split_documents(docs)

    client = chromadb.HttpClient(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        settings=Settings(allow_reset=os.getenv("DB_ALLOW_RESET"))
    )

    if args.reset:
        client.reset()  # resets the database

    collection = client.create_collection(
        name=collection
    )

    for doc in docs_splitted:
        collection.add(
            ids=[str(uuid.uuid1())], 
            metadatas=doc.metadata,
            documents=doc.page_content
        )

if __name__ == "__main__":
    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)

    parser = argparse.ArgumentParser(
        prog='StoreVectors',
        description='Stores embedded vectors to ChromaDB'
    )
    parser.add_argument(
        '-p', 
        '--path', 
        type=dir_path
    )
    parser.add_argument(
        '--reset',
        action='store_true'
    )
    parser.add_argument(
        '-c', 
        '--collection', 
        type=str, 
        default="default"
    )

    args = parser.parse_args()

    main(
        dir_path=args.path,
        collection=args.collection,
        reset=args.reset
    )