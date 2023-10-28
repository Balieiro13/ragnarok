import os
import typer
from dotenv import load_dotenv

from langchain.vectorstores.chroma import Chroma
from typing_extensions import Annotated

from knowledge_base.config import KBConfig
from chain.setup import get_llm, runnable_chain


load_dotenv()
# app = typer.Typer()

# @app.command()
def main(
    question: str,
    cn: str = "pf2e",
    k: int = 5,
    verbose: bool =False, 
    openai: bool = False,
    openllm: bool = False
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
        embedding_fn_kwargs={
            "model_name": os.getenv("EMBEDDING_MODEL_NAME"),
            "device": os.getenv("EMBEDDING_DEVICE"),
            "normalize_embeddings": False
        }
    )
    retriever = Chroma(
        client=db_config.client,
        collection_name=cn,
        embedding_function=db_config.embedding_fn,
    ).as_retriever(
        search_type="mmr",
        search_kwargs={'k': k, 'fetch_k': 40}
    )

    if openai:
        llm = get_llm(
            "openai", 
            temperature=0.6,
            model_name=os.getenv("OPENAI_API_MODEL"),
            openai_key=os.getenv("OPENAI_API_KEY")
        )
    elif openllm:
        llm_kwargs = {
            "use_llama2_prompt": False,
            "max_new_tokens":2048,
            "do_sample":True,
            "temperature":0.6,
            "top_p":0.98,
            "top_k":15,

        }
        llm = get_llm(
            llm_type="openllm",
            server_url=os.getenv("LLM_SERVER"),
            llm_kwargs=llm_kwargs
        )

    else:
        llm = get_llm(
            "hf",
            model_name_or_path=os.getenv("HF_MODEL"),
            model_kwargs={
                "device_map":"cuda",
                "trust_remote_code":False, 
                "revision":"main"
            },
            pipe_kwargs={
                "max_new_tokens":256,
                "do_sample":True,
                "temperature":0.4,
                "top_p":0.95,
                "top_k":20,
                "repetition_penalty":1.1
                
            }
        )
        

    chain = runnable_chain(llm, default_template, retriever)
    response = chain.invoke(question)
    
    print(response)

if __name__=="__main__":
    # app()
    typer.run(main)
