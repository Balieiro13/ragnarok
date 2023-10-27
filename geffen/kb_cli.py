import os
import typer
from dotenv import load_dotenv
from typing_extensions import Annotated


from knowledge_base.config import KBConfig
from knowledge_base.repository import KBRepository
from knowledge_base.etl import KBETL

load_dotenv()

app = typer.Typer()
collection_app = typer.Typer()
app.add_typer(collection_app, name="collection")


DB_CONFIG = KBConfig(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    embedding_fn_kwargs={
        "model_name": os.getenv("EMBEDDING_MODEL_NAME"),
        "device": os.getenv("EMBEDDING_DEVICE"),
        "normalize_embeddings": False
    }
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

@collection_app.command()
def query(
    query: str,
    collection_name: str = "default",
    k: int = 5,
) -> None:
    response = CLIENT.query(collection_name, query, k)
    print(response)

@app.command("store")
def store_vectors(
    path: str,
    collection_name: str = "default",
    chunk_size: int = 300, 
    chunk_overlap: int = 20, 
) -> None:
    collection = CLIENT.get_or_create_collection(collection_name)
    etl = KBETL(collection)

    print("Loading data from directory...")
    etl.extract_data(path)
    etl.split_text(chunk_size=chunk_size, 
                   chunk_overlap=chunk_overlap)

    print("Embedding data on the VectorStore...")
    etl.embed_data()
    print("Done!")   

@app.command("reset")
def reset_chroma(
    force: Annotated[
        bool, typer.Option(prompt="Are you sure you want to delete the user?")
    ],
):
    print("Reseting database...")
    CLIENT.reset()
    print("Done!")   
    

if __name__ == "__main__":
    app()
