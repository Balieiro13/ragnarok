from typing import List 

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader


def pdf_loader(dir_path:str, **kwargs) -> List[Document]:
    loader = DirectoryLoader(dir_path,
                             show_progress=True,
                             use_multithreading=True,
                             loader_cls=PyPDFLoader
                             **kwargs)
    return loader.load()

def split_data_in_chunks(
    data:List,
    chunk_size:int, 
    chunk_overlap:int,
) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)

    return chunks