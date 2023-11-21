import os
import typer
from dotenv import load_dotenv

from langchain.vectorstores.chroma import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference

from knowledge_base.config import KBConfig
from knowledge_base.embeddings.embedding_functions import HFTEIEmbeddingFunction
from chain.retrieval import retrieval_qa


def main(
    request: str,
    cn: str = "pf2e",
    k: int = 10,
    temp: float = 0.6,
    max_tokens: int = 1024
) -> None:

    zephyr_template = '''<|system|> 
    You are Geffen, a helpful AI assistant that gives a \
    response to a request based on the following context. \
    Only return the response and nothing more. 
    Context: {context}</s>
    <|user|>
    Request: {request}</s>
    <|assistant|>
    Response:'''

    db_config = KBConfig(
        host=os.getenv("DB_HOST"),
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
        search_kwargs={'k': k, 
                       'fetch_k': int(2*k),
                       'lambda_mult': 0.85}
    )
    llm = HuggingFaceTextGenInference(
        inference_server_url=os.getenv("LLM_SERVER"),
        max_new_tokens=max_tokens,
        do_sample=True,
        top_k=10,
        top_p=0.95,
        temperature=temp,
        repetition_penalty=1.1,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    chain = retrieval_qa(llm, zephyr_template, retriever)
    chain.invoke(request)
    print()
    
if __name__=="__main__":
    import warnings
    warnings.filterwarnings("ignore")
    load_dotenv()
    typer.run(main)
