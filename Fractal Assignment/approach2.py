from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import openai
import os
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
# openai.api_key = os.getenv("key")
key = os.getenv('key')

os.environ['OPENAI_API_KEY'] = key

loader = DirectoryLoader('./', glob='*.pdf')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=key)
docsearch = Chroma.from_documents(texts, embeddings)

qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type='stuff', vectorstore=docsearch,
                                return_source_documents=True)

query = 'In which NAIC member state model adoption is still pending?'
# answer = qa.run(query)
# print(answer)

result = qa({'query': query})
print(result)

