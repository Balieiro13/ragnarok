from typing import Dict, Any

from langchain.embeddings import HuggingFaceEmbeddings


def get_embedding_model(
    model_name:str, 
    model_kwargs:Dict[str, Any]={"device": "cuda"}, 
    encode_kwargs:Dict[str, Any]={"normalize_embeddings": False}
    ) -> HuggingFaceEmbeddings:

    embedding_model = HuggingFaceEmbeddings(model_name=model_name,
                                         model_kwargs=model_kwargs,
                                         encode_kwargs=encode_kwargs)

    return embedding_model
