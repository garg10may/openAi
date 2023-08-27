from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter,
)
from langchain import OpenAI, VectorDBQA, HuggingFaceHub, LLMChain
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
import openai
import os
from dotenv import find_dotenv, load_dotenv
import pickle
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    AutoModelForSeq2SeqLM,
)
import transformers
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders.parsers import GrobidParser
from langchain.document_loaders.generic import GenericLoader


def load_key():
    _ = load_dotenv(find_dotenv())

    openai_key = os.getenv("openai_key")
    os.environ["OPENAI_API_KEY"] = openai_key

    huggingface_key = os.getenv("huggingface_key")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_key


def split_documents(loader="pypdf"):
    if loader == "pypdf":
        loader = DirectoryLoader(
            "./", glob="*.pdf", loader_cls=PyPDFLoader
        )  # better than textloader, without PyPDFLoader fails miserable where there are tables or formatted text, not just simple paragraph, but still not the best
        documents = loader.load()

    if loader == "grobid":
        loader = GenericLoader.from_filesystem(
            "./", glob="*.pdf", parser=GrobidParser(segment_sentences=False)
        )
        documents = loader.load()

    # text_splitter = CharacterTextSplitter(
        # chunk_size=300, chunk_overlap=20
    # )  # default chunk_size is 1000, and chunk_overlap is 200

    text_splitter = (
    RecursiveCharacterTextSplitter()
    )  # more or less same performance as characterTextSplitter, need to check on more refined questions

    texts = text_splitter.split_documents(documents)
    # text_splitter = NLTKTextSplitter()
    # texts = text_splitter.split_text(documents)
    return texts


def get_llm(type="openai"):
    if type == "openai":
        llm = OpenAI()  # required API and dough

    if type == "huggingface":
        model_kwargs = {"temperature": 0, "max_length": 64}
        model_id = "google/flan-t5-xl"  # times out with free api
        model_id = "google/flan-t5-base"  # not working, gives poor answer
        # model_id = 'google/flan-t5-small'
        llm = HuggingFaceHub(repo_id=model_id)

    if type == "local":
        model_id = "google/flan-t5-xl"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")

        pipeline = transformers.pipeline(
            "text2text-generation", model=model, tokenizer=tokenizer
        )
        llm = HuggingFacePipeline(pipeline=pipeline)

    return llm


def get_embeddings_model(type):
    if type == "openai":
        embeddings_model = OpenAIEmbeddings()
    if type == "huggingface":
        model_name = "hkunlp/instructor-xl"
        # model_kwargs = {'device': 'cuda'}
        embeddings_model = HuggingFaceInstructEmbeddings(model_name=model_name)

    return embeddings_model


def generate_vectors(embeddings_model, texts, use_pregenerated_embeddings=False):
    persist_dir = "MyTextEmbedding"
    if use_pregenerated_embeddings:
        vectordb = Chroma(
            persist_directory=persist_dir, embedding_function=embeddings_model
        )
    else:  # generate new and also save them if required for future
        vectordb = Chroma.from_documents(
            texts, embeddings_model, persist_directory=persist_dir
        )
        # vectordb = FAISS.from_documents(texts, embeddings_model) #FAISS is just a library
        vectordb.persist()
    return vectordb


def generate_answer(vectordb, query, llm):
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectordb.as_retriever()
    )
    answer = chain.run(query)
    return answer


def filter_relevant_vectors(embedding_model, vectordb, query):
    filtered_docs = vectordb.similarity_search(query)
    vectordb = Chroma.from_documents(documents=filtered_docs, embedding=embedding_model)
    return vectordb


def get_conversation_chain(vectordb, llm):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectordb.as_retriever, memory=memory
    )
    return conversation_chain


# TEST_QUERIES = [
    # "What is the purpose of the act?",
    # "How many different sections are there? Give me brief summary of each of the section.",
    # "Which NAIC member states have still not implemented the model and which has implemented the same?",  # hugging embedding has been best here, but none have been fully correct
    # "Give headings of Section 1, Section 2, Section 3, Section 4, Section 5, Section 6",  # fails
    # "Please give me the table of contents for this pdf?",  # fails #says I don't know, till now nobody has answered it
# 
# ]


TEST_QUERIES = [
    'Summarize me the story'
]

def test_model(vectordb, llm):
    for query in TEST_QUERIES:
        answer = generate_answer(vectordb, query, llm)
        print(query)
        print('\n')
        print(answer)
        print("-" * 90)
        # dateString = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open('result1.csv', "a") as f:
            f.write('\n')
            f.write('Question: ' + query)
            f.write('\n')
            f.write('Ans. :' + answer)
            f.write("-" * 90)

def setup():
    load_key()
    texts = split_documents(loader="pypdf")  # default is pypdf
    # embeddings_model = get_embeddings_model('openai')
    embeddings_model = get_embeddings_model("huggingface")
    vectordb = generate_vectors(embeddings_model, texts, use_pregenerated_embeddings=True)
    # vectordb = filter_relevant_vectors(embeddings_model, vectordb, query) #to save costs, just submit relevant vector to reduce token
    llm = get_llm()  # default openai, other: huggingface, local
    # test_model(vectordb, llm)
    query = 'What is this document about?'
    answer = generate_answer(vectordb, query, llm)
    print(answer)



if __name__ == "__main__":
    setup()
