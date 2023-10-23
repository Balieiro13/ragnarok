import os
import argparse

from dotenv import load_dotenv

from db.manage import ChromaControl
from chain.setup import setup_chain, get_llm


load_dotenv()

def main(
    question:str,
    collection_name:str,
    verbose=False, 
    ) -> None:

    default_template = '''
    System: You are an assistant that answers questions based on a given context. 
    Use the following context to answer the question.
    If you don't know the answer, just say that you don't know. 

    Context: {context}

    Question: {question}

    Only return the helpful answer below and nothing else:

    helpful answer
    '''

    db = ChromaControl(
        server_host = os.getenv("DB_HOST"),
        server_port = os.getenv("DB_PORT"),
    )

    db.set_embedding_function(
        model_name=os.getenv("EMBEDDING_MODEL_NAME"),
        device=os.getenv("EMBEDDING_DEVICE"),
        normalize_embeddings=False
    )

    llm = get_llm(os.getenv("LLM_SERVER"))
    context = db.query(collection_name, question)

    chain = setup_chain(template=default_template, 
                        llm=llm, verbose=verbose)
    response = chain.run(context=context["documents"], 
                         question=question)
    print(response)

if __name__=="__main__":
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
    parser.add_argument(
        '-v', 
        '--verbose', 
        action='store_true'
    )
    args = parser.parse_args()

    main(
        question=args.question,
        collection_name=args.collection,
        verbose=args.verbose,
    ) 
