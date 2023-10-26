import os
import argparse
from dotenv import load_dotenv
from typing import Dict, Optional

from chroma.config import ChromaConfig
from chroma.repository import ChromaRepository
from chroma.etl import ChromaETL

load_dotenv()


DB_CONFIG = ChromaConfig(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    embedding_fn_kwargs={
        "model_name": os.getenv("EMBEDDING_MODEL_NAME"),
        "device": os.getenv("EMBEDDING_DEVICE"),
        "normalize_embeddings": False
    }
)

CLIENT = ChromaRepository(
    client=DB_CONFIG.client,
    embedding_fn=DB_CONFIG.embedding_fn
)

def list_collections() -> None:
    print(CLIENT.list_collections())

def delete_collection(collection_name: str) -> None:
    print("Deleting collection...")
    CLIENT.delete_collection(collection_name)
    print("Done!")

def store_vectors(
    dir_path: str,
    collection_name: str,
    chunk_size: int, 
    chunk_overlap: int, 
) -> None:
    collection = CLIENT.get_or_create_collection(collection_name)
    etl = ChromaETL(collection)

    print("Loading data from directory...")
    etl.extract_data(dir_path)
    etl.split_text(chunk_size=chunk_size, 
                   chunk_overlap=chunk_overlap)

    print("Embedding data on the VectorStore...")
    etl.embed_data()
    print("Done!")

def query(
    collection_name: str,
    query: str,
    k: int = 5,
    filter: Optional[Dict[str, str]] = None,
    where_document: Optional[Dict[str, str]] = None,
) -> None:
    response = CLIENT.query(collection_name, query,
                            k, filter, where_document)
    print(response)
    

def main(args):
    
    if args.action == 'collection':
        if args.list:
            list_collections() 
        if args.remove:
            delete_collection(args.remove)

    if args.action == 'store':
        store_vectors(
            args.path,
            args.collection,
            args.chunk_size,
            args.chunk_overlap
        )
    if args.action == 'query':
        #TODO
        pass

if __name__=="__main__":
    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)

    parser = argparse.ArgumentParser(
                    description="Interface with ChromaDB"
    )
    
    parser.add_argument(
        'action',
        choices=['store', 'collection', 'query']
    )
    parser.add_argument(
        '-ls',
        '--list',
        action='store_true'
    )
    parser.add_argument(
        '-rm',
        '--remove',
        type=str
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
    )
    parser.add_argument(
        '--chunk-size', 
        type=int, 
        default=300
    )
    parser.add_argument(
        '--chunk-overlap', 
        type=int, 
        default=20
    )
    args = parser.parse_args()
    main(args)
