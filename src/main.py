import argparse

from langchain.llms import OpenLLM

from setup import (
    get_db_instance,
    setup_chain,
    setup_rag_chain
)


def main(question, collection, prompt=None):
    default_template = '''
    System: You are an assistant that answers questions based on a given context. 
    Use the following context to answer the question.
    If you don't know the answer, just say that you don't know. 

    Context: {context}

    Question: {question}

    Only return the helpful answer below and nothing else:

    helpful answer
    '''

    if prompt:
        template = "".join(prompt.readlines())
    else:
        template = default_template

    retriever = get_db_instance(collection).as_retriever()

    llm = OpenLLM(
        server_url='http://localhost:3000',
        max_new_tokens=256,
        do_sample=True,
        temperature=0.5,
        top_p=0.95,
        top_k=15,
        repetition_penalty=1.18
    )

    rag_chain = setup_rag_chain(
        llm=llm,
        retriever=retriever,
        template=template
    )
    response = rag_chain.invoke(question)

    # chain = setup_chain(template=template, llm=llm)
    # context = retriever.get_relevant_documents(question)
    # response = chain.run({"context": context,
    #                       "question": question})

    print(response)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
                    prog='question',
                    description='Responds a given question using Llama2 and RAG technique'
    )
    parser.add_argument(
        '-c', 
        '--collection',
        type=str, 
        default="default"
    )
    parser.add_argument(
        '-q', 
        '--question', 
        type=str
    )
    parser.add_argument(
        '--prompt', 
        type=argparse.FileType("r")
    )

    args = parser.parse_args()

    main(
        question=args.question,
        collection=args.collection,
        prompt=args.prompt
    ) 
