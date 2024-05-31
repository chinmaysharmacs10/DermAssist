import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()


def tester(query):
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    store = LocalFileStore("./cache/")

    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

    core_embeddings_model = HuggingFaceEmbeddings(
        model_name=embed_model_id
    )

    embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model, store, namespace=embed_model_id
    )

    vector_store = FAISS.from_documents(documents=splits, embedding=embedder)
    retriever = vector_store.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain.invoke(query)


def test_gpt(query):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    llm_response = llm.invoke(query)
    str_parse = StrOutputParser()
    return str_parse.invoke(llm_response)


if __name__ == '__main__':
    input_query = "What is Task Decomposition?"
    print(test_gpt(input_query))
