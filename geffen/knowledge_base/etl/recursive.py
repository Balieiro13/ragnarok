import uuid
from tqdm import tqdm

from langchain.document_loaders import ConcurrentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from knowledge_base.etl.base import KBETL
from knowledge_base.embeddings.types import EmbeddingFunction


class KBRecursive(KBETL):
    # For now, this class has only implemented PyPDFDirectoryLoader and
    # RecursiveCharacterTextSplitter objects. In the Future, it would
    # be nice to have more flexibility with the implementation of
    # others Loaders and Splitters objects (docx, html, markdown, etc)

    def __init__(self, collection):
        self.collection = collection

    def extract_data(self, filepath: str) -> None:
        loader = ConcurrentLoader.from_filesystem(
            path=filepath,
            glob="**/*.pdf",
            suffixes=[".pdf"],
            show_progress=True,
        )
        self.documents = loader.load()
    
    def split_in_chunks(self, chunk_size: int, chunk_overlap: int) -> None:
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.chunks = splitter.split_documents(self.documents)
        self.chunks_content = [chunk.page_content for chunk in self.chunks]
        self.chunks_metadata = [chunk.metadata for chunk in self.chunks]
    
    def embed_data(self, embedding_fn: EmbeddingFunction) -> None:
        self.embeddings = embedding_fn(self.chunks_content)
        
    def load_data(self, batch_size: int = 128) -> None:
        ids = [str(uuid.uuid1()) for _ in range(len(self.chunks))] 

        for i in tqdm(range(0, len(ids), batch_size)):
            self.collection.add(
                documents=self.chunks_content[i:i+batch_size],
                embeddings=self.embeddings[i:i+batch_size],
                metadatas=self.chunks_metadata[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
