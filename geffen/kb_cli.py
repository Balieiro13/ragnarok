import os
import typer
from dotenv import load_dotenv
from typing_extensions import Annotated

from knowledge_base.config import KBConfig
from knowledge_base.repository import KBRepository
from knowledge_base.etl.recursive import KBRecursive
from knowledge_base.embeddings.embedding_functions import HFTEIEmbeddingFunction


app = typer.Typer()
collection_app = typer.Typer()
app.add_typer(collection_app, name="collection")

DB_CONFIG = KBConfig(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    embedding_fn=HFTEIEmbeddingFunction(
        os.getenv("EMBEDDING_FN_SERVER"),
        verbose=True
    )
)
CLIENT = KBRepository(
    client=DB_CONFIG.client,
    embedding_fn=DB_CONFIG.embedding_fn
)

@collection_app.command("ls")
def list_collections():
    print(CLIENT.list_collections())

@collection_app.command("rm")
def delete_collection(collection_name: str) -> None:
    print("Deleting collection...")
    CLIENT.delete_collection(collection_name)
    print("Done!")

@app.command()
def query(
    query: str,
    cn: str = "default",
    k: int = 5,
) -> None:
    response = CLIENT.query(cn, query, k)
    print(response)

@app.command("store")
def store_vectors(
    path: str,
    cn: str = "default",
    chunk_size: int = 300, 
    chunk_overlap: int = 20, 
) -> None:
    collection = CLIENT.get_or_create_collection(cn)
    etl = KBRecursive(collection)

    print("Loading data from directory...")
    etl.extract_data(path)

    print("Spliting data in Chunks...")
    etl.split_in_chunks(chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap)

    print("Embedding chunks...")
    etl.embed_data(embedding_fn=DB_CONFIG.embedding_fn)

    print("Loading data into VectorStore...")
    etl.load_data()

    print("Done!")   

@app.command("reset")
def reset_chroma(
    force: Annotated[
        bool, typer.Option(prompt="Are you sure you want to reset the database?")
    ],
) -> None:
    
    if force:
        print("Reseting database...")
        CLIENT.reset()
        print("Done!")   
    else:
        print("Operation cancelled")
    

if __name__ == "__main__":
    load_dotenv()
    app()
