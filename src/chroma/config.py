from typing import Dict, Any, Optional
from dataclasses import dataclass

from chromadb import HttpClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.config import Settings


@dataclass
class ChromaConfig:
    host: str
    port: Optional[int] = None 
    settings: Optional[Dict[str, Any]] = None
    embedding_fn_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            **self.embedding_fn_kwargs
        )
        
        if self.settings:
            settings = Settings(**self.settings)
        else:
            settings = Settings()

        if self.host.startswith("https"):
           self.client = HttpClient(host=self.host,
                                    settings=settings,
                                    ssl=True)
        else:
            self.client = HttpClient(host=self.host,
                                     port=self.port,
                                     settings=settings)
