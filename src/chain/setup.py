from langchain.llms import OpenLLM, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.retriever import BaseRetriever

def openai_llm(openai_key: str, **kwargs) -> OpenAI:
    llm = OpenAI(temperature=0.5)
    return llm
    
    
def openllm(server_url:str, **llm_kwargs) -> OpenLLM:
    llm = OpenLLM(server_url=server_url, **llm_kwargs)
    return llm

def get_llm(llm_type, **llm_kwargs) -> OpenLLM:
    if llm_type == 'openai':
        return openai_llm(**llm_kwargs)
    else:
        return openllm(**llm_kwargs)

def setup_prompt(template):
    return PromptTemplate.from_template(
        template=template
    )

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
