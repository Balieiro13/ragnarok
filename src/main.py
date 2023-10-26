import os
import argparse

from dotenv import load_dotenv

from chroma.config import ChromaConfig
from chroma.service import ChromaService
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
    db_config = ChromaConfig(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        embedding_fn_kwargs={
            "model_name": os.getenv("EMBEDDING_MODEL_NAME"),
            "device": os.getenv("EMBEDDING_DEVICE"),
            "normalize_embeddings": False
        }
    )
    db_service = ChromaService(db_config)
    context = db_service.query(collection_name, question, k=5)

    llm = get_llm(
        server_url=os.getenv("LLM_SERVER"),
        max_new_tokens=256,
        do_sample=True,
        temperature=0.5,
        top_p=0.95,
        top_k=15,
    )
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
        'question',
        type=str,
    )
    parser.add_argument(
        '-c', 
        '--collection',
        type=str, 
        default="default"
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
