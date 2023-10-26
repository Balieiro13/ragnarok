import os
import typer
from dotenv import load_dotenv
from typing import Dict, Optional

from chroma.config import ChromaConfig
from chroma.repository import ChromaRepository
from chroma.etl import ChromaETL

load_dotenv()

app = typer.Typer()
collection_app = typer.Typer()
app.add_typer(collection_app, name="collection")


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


@collection_app.command("ls")
def list_collections():
    print(CLIENT.list_collections())

@collection_app.command()
def remove(collection_name: str) -> None:
    print("Deleting collection...")
    CLIENT.delete_collection(collection_name)
    print("Done!")

@collection_app.command()
def query(
    collection_name: str,
    query: str,
    k: int = 5,
) -> None:
    response = CLIENT.query(collection_name, query, k)
    print(response)

@app.command("store")
def store_vectors(
    path: str,
    name: str = "default",
    chunk_size: int = 300, 
    chunk_overlap: int = 20, 
) -> None:
    collection = CLIENT.get_or_create_collection(name)
    etl = ChromaETL(collection)

    print("Loading data from directory...")
    etl.extract_data(path)
    etl.split_text(chunk_size=chunk_size, 
                   chunk_overlap=chunk_overlap)

    print("Embedding data on the VectorStore...")
    etl.embed_data()
    print("Done!")   

if __name__ == "__main__":
    app()
