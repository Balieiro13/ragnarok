import os
import json
import torch
import chromadb

from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


load_dotenv()


def get_db_instance(collection):
    embedding_function = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL_NAME"),
        model_kwargs=json.loads(os.getenv("EMBEDDING_MODEL_KWARGS")),
        encode_kwargs=json.loads(os.getenv("EMBEDDING_ENCODE_KWARGS")),
    )

    client = chromadb.HttpClient(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

    db = Chroma(
        client=client, 
        collection_name=collection,
        embedding_function=embedding_function
    )
    return db


def setup_llm_from_pipe():
    model_name_or_path = os.getenv("LLM_MODEL_NAME")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                device_map={"": 0},
                                                trust_remote_code=True,
                                                torch_dtype=torch.bfloat16,
                                                revision="main")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
        top_k=10,
        repetition_penalty=1.18,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return llm


def setup_chain(llm, template):
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    return chain

