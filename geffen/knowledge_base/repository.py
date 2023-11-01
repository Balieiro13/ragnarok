from typing import List, Dict, Optional
from chromadb.api.types import Document

from chromadb import HttpClient
from chromadb.types import Collection

from knowledge_base.embeddings.types import EmbeddingFunction


class KBRepository:
    
    def __init__(self, client: HttpClient, embedding_fn: EmbeddingFunction) -> None:
        self.client = client
        self.embedding_function = embedding_fn

    def create_collection(self, collection_name: str) -> None:
        self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def get_collection(self, collection_name: str) -> Collection:
        return self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_function 
        )

    def get_or_create_collection(self, collection_name: str) -> Collection:
        return self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function 
        )

    def list_collections(self) -> List[Collection]:
        return self.client.list_collections()

    def update_collection(self, collection_name: str) -> None:
        #TODO
        pass
    
    def delete_collection(self, collection_name: str) -> None:
        self.client.delete_collection(
            name=collection_name
        )

    def reset(self):
        self.client.reset()

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