from typing import Dict, Any, List

from knowledge_base.embeddings.types import EmbeddingFunction, Documents, Embeddings

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    # Since we do dynamic imports we have to type this as Any
    models: Dict[str, Any] = {}

    # If you have a beefier machine, try "gtr-t5-large".
    # for a full list of options: https://huggingface.co/sentence-transformers, https://www.sbert.net/docs/pretrained_models.html
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
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
        return self._model.encode(  # type: ignore
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