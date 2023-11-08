from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from chromadb import HttpClient
from chromadb.config import Settings

from knowledge_base.embeddings.types import EmbeddingFunction


@dataclass
class KBConfig:
    embedding_fn: EmbeddingFunction
    host: str
    port: Optional[int] = None 
    settings: Optional[Dict[str, Any]] = None

    @property
    def client(self):
        if self.settings:
            settings = Settings(**self.settings)
        else:
            settings = Settings()

        if self.host.startswith("http"):
           return HttpClient(host=self.host,
                             settings=settings,
                             ssl=False)
        else:
            return HttpClient(host=self.host,
                              port=self.port,
                              settings=settings)
