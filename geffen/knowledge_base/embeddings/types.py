from typing import Dict, Protocol, List

Embedding = List[float]
Embeddings = List[Embedding]

Metadata = Dict[str, str]
Metadatas = List[Metadata]

Document = str
Documents = List[Document]


class EmbeddingFunction(Protocol):
    def __call__(self, texts: Documents) -> Embeddings:
        ...