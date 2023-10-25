import os
import argparse
from dotenv import load_dotenv

from chroma.config import ChromaConfig
from chroma.etl import ChromaETL
from chroma.repository import ChromaRepository

load_dotenv()

def main(
    dir_path: str,
    collection_name: str,
    chunk_size: int, 
    chunk_overlap: int, 
    reset=False
    ) -> None:
    db_config = ChromaConfig(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        embedding_fn_kwargs={
            "model_name": os.getenv("EMBEDDING_MODEL_NAME"),
            "device": os.getenv("EMBEDDING_DEVICE"),
            "normalize_embeddings": False
        }
    )

    db = ChromaRepository(
        client=db_config.client,
        embedding_fn=db_config.embedding_fn
    )
    if reset:
        db.reset()

    collection = db.get_or_create_collection(collection_name)
    etl = ChromaETL(collection)

    print("Loading data from directory...")
    etl.load_data(dir_path)
    etl.split_text(chunk_size=chunk_size, 
                  chunk_overlap=chunk_overlap)


    print("Embedding data on the VectorStore...")
    etl.embed_data(collection_name)


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

    main(
        dir_path=args.path,
        collection_name=args.collection,
        reset=args.reset,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
