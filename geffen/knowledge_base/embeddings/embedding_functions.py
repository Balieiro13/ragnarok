from tqdm import tqdm
from typing import Dict, Any, List

from knowledge_base.embeddings.types import EmbeddingFunction, Documents, Embeddings


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

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    # Since we do dynamic imports we have to type this as Any
    models: Dict[str, Any] = {}

    # If you have a beefier machine, try "gtr-t5-large".
    # for a full list of options: https://huggingface.co/sentence-transformers, https://www.sbert.net/docs/pretrained_models.html
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cpu",
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
    ):
        self._verbose = show_progress_bar

        if model_name not in self.models:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ValueError(
                    "The sentence_transformers python package is not installed. Please install it with `pip install sentence_transformers`"
                )
            self.models[model_name] = SentenceTransformer(model_name, device=device)
        self._model = self.models[model_name]
        self._normalize_embeddings = normalize_embeddings

    def __call__(self, texts: Documents) -> Embeddings:
        return self._model.encode(
            list(texts),
            show_progress_bar=self._verbose,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
        ).tolist()
    
    def embed_documents(self, docs: Documents) -> Embeddings:
        return self.__call__(docs)
    
    def embed_query(self, query: str) -> List[float]:
        embed_query = self.__call__([query])
        return embed_query[0]
