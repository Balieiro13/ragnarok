import argparse

from setup import get_db_instance

def main(collection, question):
    retriever = get_db_instance(collection).as_retriever()
    
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
        collection=args.collection,
        question=args.question
    )

    