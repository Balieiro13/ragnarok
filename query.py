from src.chain import (
    get_db_instance,
    setup_llm_from_pipe,
    setup_chain
)
from src.utils import args_parsed

def main():
    args = args_parsed.question_args()

    default_template = '''
    you are an assistant that answers questions based in a given context. use the following context to answer the question.
    if you don't know the answer, just say that you don't know. use three sentences maximum and keep the answer concise.

    context: {context}

    question: {question}

    only return the helpful answer below and nothing else:

    helpful answer
    '''

    if args.prompt:
        template = "".join(args.prompt.readlines())
    else:
        template = default_template

    db = get_db_instance(args.collection)
    retriever = db.as_retriever()
    llm = setup_llm_from_pipe()
    chain = setup_chain(template=template, llm=llm)

    question = args.question
    context = retriever.get_relevant_documents(question)
    response = chain.run({"context": context,
                          "question": question})
    print(response)

if __name__=="__main__":
    main()
