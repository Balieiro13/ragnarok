from typing import List

from chromadb import HttpClient
from chromadb.types import Collection


class ChromaRepository:
    
    def __init__(self, client: HttpClient, embedding_fn) -> None:
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
