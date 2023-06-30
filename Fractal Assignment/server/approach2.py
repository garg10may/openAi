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
import faiss
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


def load_key():
    _ = load_dotenv(find_dotenv())

    openai_key = os.getenv("openai_key")
    os.environ["OPENAI_API_KEY"] = openai_key

    huggingface_key = os.getenv("huggingface_key")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_key


def split_documents():
    loader = DirectoryLoader(
        "./", glob="*.pdf", loader_cls=PyPDFLoader
    )  # without PyPDFLoader fails miserable where there are tables or formatted text, not just simple paragraph
    documents = loader.load()
    # print(documents)

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )  # default chunk_size is 1000, and chunk_overlap is 200

    text_splitter = (
        RecursiveCharacterTextSplitter()
    )  # more or less same performance as characterTextSplitter, need to check on more refined questions

    texts = text_splitter.split_documents(documents)
    # text_splitter = NLTKTextSplitter()
    # texts = text_splitter.split_text(documents)
    return texts


def get_llm(type):
    if type == "openai":
        llm = OpenAI()  # required API and dough
        return llm
    if type == "huggingface":
        model_kwargs = {"temperature": 0, "max_length": 64}
        model_id = "google/flan-t5-xl"  # times out with free api
        model_id = "google/flan-t5-base"  # not working, gives poor answer
        # model_id = 'google/flan-t5-small'
        llm = HuggingFaceHub(repo_id=model_id)
        return llm
    if type == "local":
        model_id = "google/flan-t5-xl"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")

        pipeline = transformers.pipeline(
            "text2text-generation", model=model, tokenizer=tokenizer
        )
        llm = HuggingFacePipeline(pipeline=pipeline)
        return llm


def embeddings_model(type):
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


def query_vectors(vectordb, query, llm):
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectordb.as_retriever()
    )
    answer = chain.run(query)
    return answer


# def query_vectors_alternate(vectordb, query):
# VectorDBQA is going to be deprecated soon
# chain = VectorDBQA.from_chain_type(
# llm=OpenAI(), chain_type="stuff", vectorstore=vectordb
# )
# answer = chain.run(query)
# return answer


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


# result = chain({'query': query})
# print(result)

if __name__ == "__main__":
    load_key()
    texts = split_documents()

    # embeddings_model = embeddings_model('openai')
    embeddings_model = embeddings_model("huggingface")
    vectordb = generate_vectors(
        embeddings_model, texts, use_pregenerated_embeddings=True
    )

    # vectordb = generate_vectors_huggingface(embeddings_model, texts)

    query = (
        "Give me the short summary of each section. Add line breaks after each summary"
    )
    # query = "Give me summary of what this document is all about? How many different sections are there? give me brief summary of each of the section explaining what it highlights the most?"
    # query = "Which NAIC member states have still not implemented the model and which has implemented the same or there are some states that have done it partially?" #hugging has been best here, but none have been fully correct
    # query = "Give headings of Section 1, Section 2, Section 3, Section 4, Section 5, Section 6"  # fails
    # query = "Please give me the table of contents for this pdf?"  # fails #says I don't know, till now nobody has answered it
    query = "Generate a summary of about 1000 words what this pdf is about?"
    query = "Do not give short answers. "
    query = query + "What is the purpose of the act?"
    # to save costs, otherwise whole vectordb can also be submitted
    # vectordb = filter_relevant_vectors(embeddings_model, vectordb, query)

    llm = get_llm("local")
    answer = query_vectors(vectordb, query, llm)
    print(answer)

    # answer2 = query_vectors_alternate(vectordb, query) #more of less same type of answer, that is both would hallucinate, or both won't, or both would say don't know. sometimes only difference would be in wordings
    # print(answer2)
