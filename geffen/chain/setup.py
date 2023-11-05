from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.retriever import BaseRetriever


def setup_prompt(template):
    return PromptTemplate.from_template(
        template=template
    )

def get_llm(**kwargs) -> HuggingFaceTextGenInference:
    llm = HuggingFaceTextGenInference(
        **kwargs
    )
    return llm

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
