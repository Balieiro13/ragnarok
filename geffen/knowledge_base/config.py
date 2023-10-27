from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from chromadb import HttpClient, Documents, Embeddings
from chromadb.config import Settings

from knowledge_base.embeddings.sentence_transformers import SentenceTransformerEmbeddingFunction


@dataclass
class KBConfig:
    host: str
    port: Optional[int] = None 
    settings: Optional[Dict[str, Any]] = None
    embedding_fn_kwargs: Optional[Dict[str, Any]] = None

    @property
    def embedding_fn(self):
        return SentenceTransformerEmbeddingFunction(
            **self.embedding_fn_kwargs
        )

    @property
    def client(self):
        if self.settings:
            settings = Settings(**self.settings)
        else:
            settings = Settings()

        if self.host.startswith("https"):
           return HttpClient(host=self.host,
                             settings=settings,
                             ssl=True)
        else:
            return HttpClient(host=self.host,
                              port=self.port,
                              settings=settings)
