from langchain.llms import OpenLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def get_llm(server_url:str, **llm_kwargs) -> OpenLLM:
    llm = OpenLLM(server_url=server_url, **llm_kwargs)
    return llm

def setup_chain(llm, template, **kwargs):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt, **kwargs)
    return chain
