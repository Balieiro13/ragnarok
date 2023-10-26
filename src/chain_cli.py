import os
import typer

from dotenv import load_dotenv

from chroma.config import ChromaConfig
from chroma.repository import ChromaRepository
from chain.setup import setup_chain, get_llm


load_dotenv()

def main(
    question: str,
    collection_name :str = 'default',
    verbose: bool =False, 
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
    db_repo = ChromaRepository(
        client=db_config.client,
        embedding_fn=db_config.embedding_fn
    )
    context = db_repo.query(collection_name, question, k=5)

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
    typer.run(main)