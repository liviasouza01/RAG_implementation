import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore #vector store... mas poderiamos usar outros como ChromaDB, Weviate, etc...

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("mediumblog1.txt") #doc de exemplo
    documents = loader.load() #mesma implementação para outros docs (slack, youtube...)
    
    print("splitting...")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) #chunk_size é uma heurística... deve ser grande o suficiente para que o humano possa entender o contexto do doc
                                                                            #chunk_overlap é o número de caracteres que serão sobrepostos entre chunks, a informação de contexto que será perdida
    texts = text_splitter.split_documents(documents)
    print(f"doc splitted into {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get('OPENAI_API_KEY'))

    print("embedding...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    print("done")