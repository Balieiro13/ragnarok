import os
import typer
from dotenv import load_dotenv

from langchain.vectorstores.chroma import Chroma
from typing_extensions import Annotated

from knowledge_base.config import KBConfig
from knowledge_base.embeddings.embedding_functions import SentenceTransformerEmbeddingFunction
from chain.setup import get_llm, runnable_chain, hftgi_llm


load_dotenv()
# app = typer.Typer()

# @app.command()
def main(
    question: str,
    cn: str = "pf2e",
    k: int = 10,
    temp: float = 0.8,
    max_tokens: int = 256,
    verbose: bool = False, 
    openai: bool = False,
) -> None:

    default_template = '''
    You are an assistant that answers a request based on the following context.
    Think about the informations that the context gives and return the most helpful aswer.

    Context: {context}

    User Request: {request}

    Assistant Helpful answer: 
    '''
    embedding_fn_kwargs={
        "model_name": os.getenv("EMBEDDING_MODEL_NAME"),
        "device": os.getenv("EMBEDDING_DEVICE"),
        "normalize_embeddings": False
    }

    db_config = KBConfig(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        embedding_fn=SentenceTransformerEmbeddingFunction(**embedding_fn_kwargs)
    )

    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS"))
    
    retriever = Chroma(
        client=db_config.client,
        collection_name=cn,
        embedding_function=db_config.embedding_fn,
    ).as_retriever(
        search_type="mmr",
        search_kwargs={'k': k, 'fetch_k': int(2*k), 'lambda_mult': 0.75}
    )

    if openai:
        llm = get_llm(
            "openai", 
            temperature=temp,
            model_name=os.getenv("OPENAI_API_MODEL"),
            openai_key=os.getenv("OPENAI_API_KEY")
        )

    else:
        # llm = get_llm(
        #     "hf",
        #     model_name_or_path=os.getenv("HF_MODEL"),
        #     model_kwargs={
        #         "device_map":"cuda",
        #         "trust_remote_code":False, 
        #         "revision":"main"
        #     },
        #     pipe_kwargs={
        #         "max_new_tokens":MAX_NEW_TOKENS,
        #         "do_sample":True,
        #         "temperature":temp,
        #         "top_p":0.95,
        #         "top_k":15,
        #         "repetition_penalty":1.1
                
        #     }
        # )

        llm = hftgi_llm(
            inference_server_url=os.getenv("LLM_SERVER"),
            max_new_tokens=min(max_tokens,MAX_NEW_TOKENS),
            do_sample=True,
            top_k=10,
            top_p=0.95,
            temperature=temp,
            repetition_penalty=1.15,

        )

    chain = runnable_chain(llm, default_template, retriever)
    response = chain.invoke(question)
    
    print(response)

if __name__=="__main__":
    # app()
    typer.run(main)
