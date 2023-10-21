import uuid
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from chromadb import HttpClient
from chromadb.types import Collection
from chromadb.api.types import Document
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.config import Settings

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ChromaControl:
    
    def __init__(
        self, 
        server_host: str, 
        server_port: Optional[int] = None, 
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config
        self.client = self.set_client(server_host,
                                      server_port,
                                      config)

    @staticmethod
    def set_client(
        server_host:str, 
        server_port: int, 
        config: Optional[Dict[str, Any]] = None
    ):
        if config:
            settings = Settings(**config)
        else:
            settings = Settings()

        if server_host.startswith("http"):
            client = HttpClient(host=server_host,
                                settings=settings)
        else:
            client = HttpClient(host=server_host,
                                port=server_port,
                                settings=settings)
        return client 

    def reset_chroma(self):
        if self.config["allow_reset"]:
            self.client.reset()

    def set_embedding_function(
        self, 
        model_name: str, 
        device: str,
        normalize_embeddings: bool = False,
    ) -> None:
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings
        )

    def create_collection(self, collection_name: str) -> None:
        self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def delete_collection(self, collection_name: str) -> None:
        self.client.delete_collection(
            name=collection_name
        )

    def get_collection(self, collection_name: str) -> Collection:
        return self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_function 
        )

    def load_data(
        self,
        filepath: str, 
    ) -> None:
        loader = PyPDFDirectoryLoader(filepath)
        self.documents = loader.load()
    
    def split_text(self, chunk_size: int, chunk_overlap: int):
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.chunks = splitter.split_documents(self.documents)
    
    def embed_data(self, collection_name: str):
        collection = self.get_collection(collection_name)
        for chunk in tqdm(self.chunks):
            collection.add(
                ids=[str(uuid.uuid1())], 
                metadatas=chunk.metadata,
                documents=chunk.page_content,
            )

    def query(
        self,
        collection_name: str,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        
        collection = self.get_collection(collection_name)
        return collection.query(
            query_texts=[query],
            n_results=k,
            where=filter,
            where_document=where_document,
        )

