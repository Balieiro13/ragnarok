import os
import uuid
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings

from utils.data_loader import pdf_loader, split_data_in_chunks
from utils.embed import get_embedding_model

load_dotenv()

def main(dir_path, collection_name, reset=False):

    embedding_model = get_embedding_model(
        model_name=os.getenv("EMBEDDING_MODEL_NAME"),
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": False}
    )
    
    print("\nSplitting docs into chunks\n")
    docs = pdf_loader(dir_path=dir_path,
                      glob="*.pdf")

    chunks = split_data_in_chunks(data=docs,
                                  chunk_size=300,
                                  chunk_overlap=20)

    client = chromadb.HttpClient(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        settings=Settings(allow_reset=os.getenv("DB_ALLOW_RESET"))
    )

    if reset:
        client.reset()  # resets the database

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_model.embed_documents
    )

    print("\nEmbedding Chunks\n")
    for chunk in tqdm(chunks):
        collection.add(
            ids=[str(uuid.uuid1())], 
            metadatas=chunk.metadata,
            documents=chunk.page_content
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
        collection_name=args.collection,
        reset=args.reset
    )
