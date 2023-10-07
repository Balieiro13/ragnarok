import torch
import chromadb

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils.args_parsed import question_args


args = question_args()

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embedding_function = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

client = chromadb.HttpClient(
    host='localhost',
    port=8000, 
)

db = Chroma(
    client=client, 
    collection_name=args.collection,
    embedding_function=embedding_function
)

retriever = db.as_retriever()


model_name_or_path = "TheBloke/Llama-2-7B-GPTQ"
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

defatul_template = """
You are an assistant that gives one answer for a question. Use the following context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Context: {context} 

Question: {question}

Only return the helpful answer below and nothing else
Helpful answer

"""
if args.prompt:
    template = "".join(args.prompt.readlines())
else:
    template = defatul_template

print(template)

prompt = PromptTemplate(
    input_variables=['context', 'question'],
    template=template,
)

chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

question = args.question
context = retriever.get_relevant_documents(question)
response = chain.run({"context": context,
                      "question": question})
print(response)
