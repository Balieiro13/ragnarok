#!/usr/bin/env python
import os
from dotenv import load_dotenv

from langchain.vectorstores.chroma import Chroma
from langchain.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from knowledge_base.config import KBConfig
from knowledge_base.embeddings.embedding_functions import HFTEIEmbeddingFunction
from chain.retrieval import retrieval_qa

from typing import List, Tuple

from fastapi import FastAPI
from langchain.chains import ConversationalRetrievalChain
from langchain.pydantic_v1 import BaseModel, Field

from langserve import add_routes

load_dotenv()
zephyr_template = '''<|system|> You are Geffen, a helpful AI assistant that gives a response to a request based on the following context. Only return the response and nothing more. 
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
        collection_name="pf2e",
        embedding_function=db_config.embedding_fn,
    ).as_retriever(
        search_type="mmr",
        search_kwargs={'k': 10, 
                       'fetch_k': int(2*10),
                       'lambda_mult': 0.85}
    )
llm = HuggingFaceTextGenInference(
        inference_server_url=os.getenv("LLM_SERVER"),
        max_new_tokens=256,
        do_sample=True,
        top_k=10,
        top_p=0.95,
        temperature=0.4,
        repetition_penalty=1.1,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
chain = retrieval_qa(llm, zephyr_template, retriever)


# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question", "output": "answer"}},
    )
    question: str


chain = ConversationalRetrievalChain.from_llm(llm, retriever).with_types(
    input_type=ChatHistory
) 

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, chain)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001)
