import os
import argparse
from dotenv import load_dotenv

from utils.db import ChromaControl

load_dotenv()

def main(dir_path, collection_name, reset=False):
    db = ChromaControl(
        server_host = os.getenv("DB_HOST"),
        server_port = os.getenv("DB_PORT"),
    )

    db.set_embedding_function(
        model_name=os.getenv("EMBEDDING_MODEL_NAME"),
        device="cuda",
        normalize_embeddings=False
    )

    db.create_collection(collection_name)

    print("Loading data from directory...")
    db.load_data(dir_path)
    db.split_text(chunk_size=300, chunk_overlap=20)


    print("Embedding data on the VectorStore...")
    db.embed_data(collection_name)


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
