import os
import typer

from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma

from knowledge_base.config import KBConfig
from chain.setup import llm_chain, get_llm, runnable_chain


load_dotenv()
# app = typer.Typer()

# @app.command()
def main(
    question: str,
    collection_name :str = 'default',
    instruction: str = '',
    k: int = 5,
    verbose: bool =False, 
    model: str = 'openllm'
) -> None:

    default_template = '''
    You are an assistant that answers a request based on the following context.
    If you don't know the answer, just say that you don't know. 

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
        embedding_function=db_config.embedding_fn
    ).as_retriever()

    if model == 'openllm':
        llm = get_llm(
            llm_type="openllm",
            server_url=os.getenv("LLM_SERVER"),
            max_new_tokens=256,
            do_sample=True,
            temperature=0.5,
            top_p=0.95,
            top_k=15,
        )
    if model == 'openai':
        llm = get_llm(
            "openai", 
            model_name=os.getenv("OPENAI_API_MODEL"),
            openai_key=os.getenv("OPENAI_API_KEY")
        )

    # context = retriever.get_relevant_documents(query=question)

    # chain = llm_chain(template=default_template, 
    #                     llm=llm, verbose=verbose)

    # response = chain.run(context=context, #["documents"], 
    #                      question=question, 
    #                      instructions=instruction)

    chain = runnable_chain(llm, default_template, retriever)
    response = chain.invoke(question)
    
    print(response)

if __name__=="__main__":
    # app()
    typer.run(main)