from tqdm import tqdm
from typing import List

from knowledge_base.embeddings.types import (
    EmbeddingFunction, 
    Documents, 
    Embeddings,
)


class HFTEIEmbeddingFunction(EmbeddingFunction):
    # Since we do dynamic imports we have to type this as Any
    def __init__(
        self,
        model_server: str = "http://localhost:8081/embed",
        verbose: bool = False,
    ):
        import requests

        self._url = model_server
        self._session = requests.Session()
        self._verbose = verbose


    def __call__(self, texts: Documents) -> Embeddings:
        chunk_size: int = 32
        embeddings = list()

        if not self._verbose:
            from functools import partialmethod
            tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

        for i in tqdm(range(0, len(texts), chunk_size)):
            embeddings += (self._session.post(
                self._url, json={"inputs": texts[i:i+chunk_size]}
            ).json())
        return embeddings

    def embed_documents(self, docs: Documents) -> Embeddings:
        return self.__call__(docs)

    def embed_query(self, query: str) -> List[float]:
        embed_query = self.__call__([query])
        return embed_query[0]
