import os
import typer
from dotenv import load_dotenv

from langchain.vectorstores.chroma import Chroma
from typing_extensions import Annotated

from knowledge_base.config import KBConfig
from knowledge_base.embeddings.embedding_functions import HFTEIEmbeddingFunction
from chain.setup import get_llm, runnable_chain, hftgi_llm


load_dotenv()
# app = typer.Typer()

# @app.command()
def main(
    question: str,
    cn: str = "pf2e",
    k: int = 10,
    temp: float = 0.4,
    verbose: bool =False, 
    openai: bool = False,
    openllm: bool = False,
) -> None:

    default_template = '''
    You are an assistant that answers a request based on the following context.
    Think about the informations that the context gives and return the most helpful aswer.

    Context: {context}

    Request: {request}

    Helpful answer: 
    '''
    db_config = KBConfig(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        embedding_fn=HFTEIEmbeddingFunction(
            os.getenv("EMBEDDING_FN_SERVER")
        )
    )

    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS"))
    
    retriever = Chroma(
        client=db_config.client,
        collection_name=cn,
        embedding_function=db_config.embedding_fn,
    ).as_retriever(
        search_type="mmr",
        search_kwargs={'k': k, 'fetch_k': 50, 'lambda_mult': 0.85}
    )

    llm = get_llm(
        llm_type="hftgi",
        llm_kwargs={"server_url":os.getenv("LLM_SERVER")},
    )

    chain = runnable_chain(llm, default_template, retriever)
    response = chain.invoke(question)
    
    print(response)

if __name__=="__main__":
    # app()
    typer.run(main)
