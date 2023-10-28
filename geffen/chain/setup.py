from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms.openllm import OpenLLM
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.retriever import BaseRetriever
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def setup_prompt(template):
    return PromptTemplate.from_template(
        template=template
    )


def openai_llm(openai_key: str, **kwargs) -> ChatOpenAI:
    llm = ChatOpenAI(temperature=0.6, **kwargs)
    return llm
    
def openllm(server_url:str, **llm_kwargs) -> OpenLLM:
    llm = OpenLLM(server_url=server_url, **llm_kwargs)
    return llm

def hf_llm(model_name_or_path, model_kwargs, pipe_kwargs) -> HuggingFacePipeline:
    import warnings
    warnings.filterwarnings("ignore")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **pipe_kwargs
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def get_llm(llm_type, **llm_kwargs) -> OpenLLM:
    if llm_type == 'openai':
        return openai_llm(**llm_kwargs)
    elif llm_type == 'hf':
        return hf_llm(**llm_kwargs)
    else:
        return openllm(**llm_kwargs)

def llm_chain(llm, template, **kwargs):
    prompt = setup_prompt(template)

    chain = LLMChain(llm=llm, prompt=prompt, **kwargs)
    return chain

def runnable_chain(llm, template: str, retriever: BaseRetriever):
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    prompt = setup_prompt(template)

    chain = (
        {"context": retriever | format_docs, 
         "request": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


