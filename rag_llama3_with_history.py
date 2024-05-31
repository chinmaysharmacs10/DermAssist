from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

urls = [
    "https://www.aad.org/public/diseases/acne/diy/adult-acne-treatment",
    "https://www.aad.org/public/diseases/a-z/ringworm-treatment",
]


def get_document_retriever():
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    store = LocalFileStore("./cache/")

    core_embeddings_model = HuggingFaceEmbeddings(
        model_name=embed_model_id
    )

    embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model, store, namespace=embed_model_id
    )

    doc_splits = text_splitter.split_documents(docs_list)
    vector_store = FAISS.from_documents(documents=doc_splits, embedding=embedder)
    retriever = vector_store.as_retriever()

    return retriever


def get_prompt():
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
            Question: {input} 
            Context: {context} 
            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
    )
    return prompt


class RAG:
    def __init__(self):
        self.llm = ChatOllama(model="llama3", temperature=0)
        self.retriever = get_document_retriever()
        self.prompt = get_prompt()
        self.chat_history = []

    def format_docs(self, documents):
        return "\n\n".join(doc.page_content for doc in documents)

    def generate_response(self, query):
        chat_history = self.chat_history
        retriever_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                (
                    "human",
                    "Given a chat history and the latest user question \
                    which might reference context in the chat history, formulate a standalone question \
                    which can be understood without the chat history. Do NOT answer the question, \
                    just reformulate it if needed and otherwise return it as is."
                ),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm, retriever=self.retriever, prompt=retriever_prompt
        )

        document_chain = create_stuff_documents_chain(llm=self.llm, prompt=self.prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        response = rag_chain.invoke({"input": query})

        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=response["answer"]))

        return response["answer"]


if __name__ == '__main__':
    rag = RAG()
    while True:
        input_query = input("Enter your question: ")
        if input_query == "stop":
            break
        print("\nLLM Response:")
        print(rag.generate_response(input_query), "\n")

    # print(rag.generate_response("what non-prescription treatments can I use to treat?"))
    # print(rag.generate_response("what do those treatments contain?"))
