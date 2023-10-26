import uuid
from tqdm import tqdm

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ChromaETL:

    def __init__(self, collection):
        self.collection = collection

    def load_data(self, filepath: str) -> None:
        loader = PyPDFDirectoryLoader(filepath)
        self.documents = loader.load()
    
    def split_text(self, chunk_size: int, chunk_overlap: int) -> None:
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.chunks = splitter.split_documents(self.documents)
    
    def embed_data(self) -> None:
        for chunk in tqdm(self.chunks):
            self.collection.add(
                ids=[str(uuid.uuid1())], 
                metadatas=chunk.metadata,
                documents=chunk.page_content,
            )