import os
import typer
from dotenv import load_dotenv

from langchain.vectorstores.chroma import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from knowledge_base.config import KBConfig
from knowledge_base.embeddings.embedding_functions import HFTEIEmbeddingFunction
from chain.setup import get_llm, runnable_chain


load_dotenv()
# app = typer.Typer()

# @app.command()
def main(
    request: str,
    cn: str = "emb",
    k: int = 10,
    temp: float = 0.4,
    max_tokens: int = 1024
) -> None:

    mistral_template = '''
    <s>[INST] You are assistant that response a request based on the following context. 
    Only return the response and nothing more.

    Context: {context}

    Request: {request}

    Assistant response:
    [/INST]'''

    db_config = KBConfig(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        embedding_fn=HFTEIEmbeddingFunction(
            os.getenv("EMBEDDING_FN_SERVER")
        )
    )

    retriever = Chroma(
        client=db_config.client,
        collection_name=cn,
        embedding_function=db_config.embedding_fn,
    ).as_retriever(
        search_type="mmr",
        search_kwargs={'k': k, 'fetch_k': int(2*k), 'lambda_mult': 0.75}
    )

    llm = get_llm(
        llm_type="hftgi",
        llm_kwargs={
            "inference_server_url":os.getenv("LLM_SERVER"),
            "max_new_tokens":max_tokens,
            "do_sample":True,
            "top_k":10,
            "top_p":0.95,
            "temperature":temp,
            "repetition_penalty":1.1,    
            "streaming":True,
            "callbacks":[StreamingStdOutCallbackHandler()]
        }
    )

    chain = runnable_chain(llm, mistral_template, retriever)
    chain.invoke(request)
    print()
    
if __name__=="__main__":
    # app()
    typer.run(main)
