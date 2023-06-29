from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import openai
import os
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv()) 
key = os.getenv('key')

os.environ['OPENAI_API_KEY'] = key

loader = DirectoryLoader('./', glob='*.pdf')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=key)
vectordb = Chroma.from_documents(texts, embeddings)

#VectorDBQA is going to be deprecated soon
# chain = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type='stuff', vectorstore=vectordb,
                                # return_source_documents=True)

chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                    chain_type='stuff',
                                    retriever=vectordb.as_retriever())

query = 'In which NAIC member state model adoption is still pending?'
answer = chain.run(query)
print(answer)

# result = chain({'query': query})
# print(result)

