from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.schema.runnable import RunnablePassthrough

import os
load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) #itera entre os docs

if __name__ == "__main__":
    print("Retrieving...")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    query = "What is pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    #result = chain.invoke(input={}) #essa parte não usa RAG, apenas testa o modelo de llm
    #print(result.content)

    vectorstore = PineconeVectorStore(index_name=os.environ['INDEX_NAME'], embedding=embeddings)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat") #prompt para o modelo
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt) #recebe lista de docs e formatar para o modelo
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke(input={"input": query})
    print(result)


    template = ''' Use the following pieces of context to answer the question ar the enr,
    If you don´t know the answer, just say that you don´t know. Don´t try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    Answer: '''

    custom_rag_prompt = PromptTemplate.from_template(template)

    #RAG com LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )
    res = rag_chain.invoke(query)
    print(res)
