from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from chromadb import HttpClient, Documents, Embeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.config import Settings


class CustomEmbeddingFn(SentenceTransformerEmbeddingFunction):
    '''For some reason, langchain embeddings needs to
    implement two functions: embed_documents() and embed_query(). 
    This is incompatible with chromadb embeddings, which implements
    a __call()__ class method. For that reason, i create this
    wrapper of the chroma embeddings and implements the two necessary
    functions that langchain needs. Now i can use this class in both frameworks
    '''
    
    def embed_documents(self, docs: Documents) -> Embeddings:
        return super().__call__(docs)
    
    def embed_query(self, query: str) -> List[float]:
        embed_query = super().__call__([query])
        return embed_query[0]


@dataclass
class ChromaConfig:
    host: str
    port: Optional[int] = None 
    settings: Optional[Dict[str, Any]] = None
    embedding_fn_kwargs: Optional[Dict[str, Any]] = None

    @property
    def embedding_fn(self):
        return CustomEmbeddingFn(
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
