from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
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

def hftgi_llm(**kwargs) -> HuggingFaceTextGenInference:
    llm = HuggingFaceTextGenInference(
        **kwargs
    )
    return llm

def get_llm(llm_type, **llm_kwargs) -> OpenLLM:
    models = {"hf": hf_llm,
              "hftgi": hftgi_llm,}
    
    llm = models[llm_type] 
    return llm(**llm_kwargs)

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
