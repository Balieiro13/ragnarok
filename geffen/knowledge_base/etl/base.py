import uuid
from tqdm import tqdm
from typing import Protocol

from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import TextSplitter

from knowledge_base.embeddings.types import EmbeddingFunction


class KBETL(Protocol):
    
    def extract_data(sefl):
        ...
    
    def split_in_chunks(self):
        ...
    
    def embed_data(self):
        ...
    
    def load_data(self):
        ...

class KBETLBase(KBETL):

    def extract_data(self, loader:BaseLoader) -> None:
        self.documents = loader.load()
    
    def split_in_chunks(self, splitter: TextSplitter) -> None:
        self.chunks          = splitter.split_documents(self.documents)
        self.chunks_content  = [chunk.page_content for chunk in self.chunks]
        self.chunks_metadata = [chunk.metadata for chunk in self.chunks]
        self.ids             = [str(uuid.uuid1()) for _ in self.chunks] 
    
    def embed_data(self, embedding_fn: EmbeddingFunction) -> None:
        self.embeddings = embedding_fn(self.chunks_content)
        
    def load_data(self, collection, batch_size: int = 128) -> None:
        for i in tqdm(range(0, len(self.ids), batch_size)):
            collection.add(
                documents=self.chunks_content[i:i+batch_size],
                embeddings=self.embeddings[i:i+batch_size],
                metadatas=self.chunks_metadata[i:i+batch_size],
                ids=self.ids[i:i+batch_size]
            )
