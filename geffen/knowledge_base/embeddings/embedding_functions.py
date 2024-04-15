from tqdm import tqdm
from typing import List

from langchain_community.embeddings import HuggingFaceHubEmbeddings

from knowledge_base.embeddings.types import (
    Documents, 
    Embeddings,
)


class HuggingFaceTEI(HuggingFaceHubEmbeddings):
    # Since we do dynamic imports we have to type this as Any
    def __call__(self, input: Documents) -> Embeddings:
        return super().embed_documents(input)
