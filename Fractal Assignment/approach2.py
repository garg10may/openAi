from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter,
)
from langchain import OpenAI, VectorDBQA, LLMChain
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
import openai
import os
from dotenv import find_dotenv, load_dotenv


def load_key():
    _ = load_dotenv(find_dotenv())

    openai_key = os.getenv("openai_key")
    os.environ["OPENAI_API_KEY"] = openai_key

    huggingface_key = os.getenv("huggingface_key")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_key


def split_documents():
    loader = DirectoryLoader("./", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    # print(documents)
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )  # default chunk_size is 1000, and chunk_overlap is 200
    texts = text_splitter.split_documents(documents)
    # text_splitter = NLTKTextSplitter()
    # texts = text_splitter.split_text(documents)
    return texts


def embeddings_model(type):
    if type == "openai":
        embeddings_model = OpenAIEmbeddings()

    if type == "huggingface":
        model_name = "sentence-transformers/all-mpnet-base-v2"
        embeddings_model = HuggingFaceEmbeddings(model_name=model_name)

    return embeddings_model


def generate_vectors(embeddings_model, texts, use_pregenerated_embeddings=False):
    persist_dir = "MyTextEmbedding"
    if use_pregenerated_embeddings:  # generate new embeddings
        vectordb = Chroma(
            persist_directory=persist_dir, embedding_function=embeddings_model
        )
    else:
        vectordb = Chroma.from_documents(
            texts, embeddings_model, persist_directory=persist_dir
        )
        vectordb.persist()
    return vectordb


def query_vectors(vectordb, query):
    chain = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=vectordb.as_retriever()
    )
    answer = chain.run(query)
    return answer


def query_vectors_alternate(vectordb, query):
    # VectorDBQA is going to be deprecated soon
    chain = VectorDBQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", vectorstore=vectordb
    )
    answer = chain.run(query)
    return answer


def filter_relevant_vectors(embedding_model, vectordb, query):
    filtered_docs = vectordb.similarity_search(query)
    vectordb = Chroma.from_documents(documents=filtered_docs, embedding=embedding_model)
    return vectordb


# result = chain({'query': query})
# print(result)

if __name__ == "__main__":
    load_key()
    texts = split_documents()
    embeddings_model = embeddings_model("openai")
    vectordb = generate_vectors(embeddings_model, texts)
    query = (
        "Give me the short summary of each section. Add line breaks after each summary"
    )
    query = "Give me summary of what this document is all about? How many different sections are there? give me brief summary of each of the section explaining what it highlights the most?"
    # query = "Which NAIC member states have still not implemented the model and which has implemented the same or there are some states that have done it partially?"
    # query = "Give headings of Section 1, Section 2, Section 3, Section 4, Section 5, Section 6"  # fails
    # query = "Please give me table of contents?"  # fails
    # query = 'Exlain in simple words what is this PDF about?'
    # to save costs, otherwise whole vectordb can also be submitted
    # vectordb = filter_relevant_vectors(embeddings_model, vectordb, query)
    answer1 = query_vectors(vectordb, query)
    print(answer1)

    # answer2 = query_vectors_alternate(vectordb, query) #more of less same type of answer, that is both would hallucinate, or both won't, or both would say don't know. sometimes only difference would be in wordings
    # print(answer2)
