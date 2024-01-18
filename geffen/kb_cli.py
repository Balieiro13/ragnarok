import os
import typer
from dotenv import load_dotenv
from typing_extensions import Annotated

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ConcurrentLoader

from knowledge_base.config import KBConfig
from knowledge_base.repository import KBRepository
from knowledge_base.etl.base import KBETLBase
from knowledge_base.embeddings.embedding_functions import HFTEIEmbeddingFunction


app = typer.Typer()
collection_app = typer.Typer()
app.add_typer(collection_app, name="collection")

@collection_app.command("ls")
def list_collections():
    print(CLIENT.list_collections())

@collection_app.command("rm")
def delete_collection(collection_name: str) -> None:
    print("Deleting collection...")
    CLIENT.delete_collection(collection_name)
    print("Done!")

@collection_app.command("rename")
def rename_collection(old_name: str, new_name: str) -> None:
    CLIENT.rename_collection(
        old_collection_name=old_name,
        new_collection_name=new_name,
    )
    print("The collection has been renamed!")

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
    loader = ConcurrentLoader.from_filesystem(
        path=path,
        glob="**/*.pdf",
        suffixes=[".pdf"],
        show_progress=True,
    )
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    collection = CLIENT.get_or_create_collection(collection_name=cn)
    etl = KBETLBase()

    print("Loading data from directory...")
    etl.extract_data(loader=loader)

    print("Spliting data in Chunks...")
    etl.split_in_chunks(splitter=splitter)

    print("Embedding chunks...")
    etl.embed_data(embedding_fn=DB_CONFIG.embedding_fn)

    print("Loading data into VectorStore...")
    etl.load_data(collection=collection)

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

    DB_CONFIG = KBConfig(
        host=os.getenv("DB_HOST"),
        embedding_fn=HFTEIEmbeddingFunction(
            os.getenv("EMBEDDING_FN_SERVER"),
            verbose=True
        )
    )
    CLIENT = KBRepository(
        client=DB_CONFIG.client,
        embedding_fn=DB_CONFIG.embedding_fn
    )
    app()
