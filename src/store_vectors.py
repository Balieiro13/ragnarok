import uuid

import chromadb
from chromadb.config import Settings

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.args_parsed import store_vectors_args


args = store_vectors_args()

loader = DirectoryLoader(
    args.path,
    glob="*.pdf", 
    show_progress=True,
    use_multithreading=True,
    loader_cls=PyPDFLoader
)

docs = loader.load()

# Spliting text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs_splitted = text_splitter.split_documents(docs)

client = chromadb.HttpClient(
    host='localhost',
    port=8000,
    settings=Settings(allow_reset=True)
)

if args.reset:
    client.reset()  # resets the database

    collection = client.create_collection(
        name="default"
    )

collection = client.create_collection(
    name=args.collection
)

for doc in docs_splitted:
    collection.add(
        ids=[str(uuid.uuid1())], 
        metadatas=doc.metadata,
        documents=doc.page_content
    )
