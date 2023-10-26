from typing import List, Dict, Optional
from chromadb.api.types import Document

from chroma.config import ChromaConfig
from chroma.repository import ChromaRepository


class ChromaService:
    
    def __init__(self, config: ChromaConfig):
        self.croma_repo = ChromaRepository(
            client=config.client,
            embedding_fn=config.embedding_fn
        )
        
    def get_collection(self, collection_name):
        return self.croma_repo.get_collection(collection_name)

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
