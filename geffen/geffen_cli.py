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
    collection_name: Annotated[str, typer.Argument(envvar="COLLECTION_NAME")] = "default",
    instruction: str = '',
    k: int = 5,
    verbose: bool =False, 
    openai: bool = False
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
        collection_name=collection_name,
        embedding_function=db_config.embedding_fn,
    ).as_retriever(
        search_type="mmr",
        search_kwargs={'k': 10, 'fetch_k': 50}
    )

    if openai:
        llm = get_llm(
            "openai", 
            model_name=os.getenv("OPENAI_API_MODEL"),
            openai_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        llm = get_llm(
            llm_type="openllm",
            server_url=os.getenv("LLM_SERVER"),
            use_llama2_prompt=False,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.3,
            top_p=0.98,
            top_k=15,
        )

    chain = runnable_chain(llm, default_template, retriever)
    response = chain.invoke(question)
    
    print(response)

if __name__=="__main__":
    # app()
    typer.run(main)