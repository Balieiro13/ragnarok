import argparse

from setup import get_chromadb_collection

def main(collection_name, question):
    retriever = get_chromadb_collection(collection_name).as_retriever()
    
    docs = retriever.get_relevant_documents(question)
    print(docs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='question',
                    description='Responds a given question using Llama2 and RAG technique'
    )
    parser.add_argument(
        '-c', 
        '--collection',
        type=str, 
        default="default"
    )
    parser.add_argument(
        '-q', 
        '--question', 
        type=str
    )
    args = parser.parse_args()

    main(
        collection_name=args.collection,
        question=args.question
    )

    