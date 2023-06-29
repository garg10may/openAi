from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import openai
import os
from dotenv import find_dotenv, load_dotenv

def load_key():
  _ = load_dotenv(find_dotenv()) 
  key = os.getenv('key')
  os.environ['OPENAI_API_KEY'] = key

def split_documents():
  loader = DirectoryLoader('./', glob='*.pdf')
  documents = loader.load()
  text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
  texts = text_splitter.split_documents(documents)
  return texts

def embeddings_model():
  return OpenAIEmbeddings()

def generate_vectors(embeddings_model, texts):
  vectordb = Chroma.from_documents(texts, embeddings_model)
  return vectordb

def query_vectors1(vectordb, query):
  #VectorDBQA is going to be deprecated soon
  chain = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type='stuff', vectorstore=vectordb)
  answer = chain.run(query)
  return answer

def query_vectors(vectordb, query):
  chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                    chain_type='stuff',
                                    retriever=vectordb.as_retriever())
  answer = chain.run(query)
  return answer

def filter_relevant_vectors(embedding_model, vectordb, query):
  filtered_docs = vectordb.similarity_search(query)
  vectordb = Chroma.from_documents(documents=filtered_docs, embedding=embedding_model)
  return vectordb

# result = chain({'query': query})
# print(result)

if __name__ == '__main__':
  key = load_key()
  texts = split_documents()
  embeddings_model = embeddings_model()
  vectordb = generate_vectors(embeddings_model, texts)
  query = 'Give me the short summary of each section. Add line breaks after each summary'
  #to save costs, otherwise whole vectordb can also be submitted
  vectordb = filter_relevant_vectors(embeddings_model, vectordb, query) 
  answer1 = query_vectors(vectordb, query)
  answer2 = query_vectors1(vectordb, query)
  print(answer1)
  print(answer2)


