from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import os
import warnings
import utils

warnings.filterwarnings('ignore')

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "DermAssist"
os.environ["LANGCHAIN_API_KEY"] = "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"

embedding_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
embedding_cache = "./cache/"


def get_document_retriever():
    urls = utils.get_doc_urls()
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    store = LocalFileStore(embedding_cache)
    core_embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_id)
    embedder = CacheBackedEmbeddings.from_bytes_store(core_embeddings_model, store, namespace=embedding_model_id)

    doc_splits = text_splitter.split_documents(docs_list)
    vector_store = FAISS.from_documents(documents=doc_splits, embedding=embedder)
    retriever = vector_store.as_retriever()

    return retriever


def get_llm_prompt():
    prompt = """You are an assistant for question-answering tasks. \
                Use the following pieces of retrieved context to answer the question. \
                If you don't know the answer, just say that you don't know. \
                Use three sentences maximum and keep the answer concise.\

                {context}"""

    llm_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    return llm_prompt


def get_retriever_prompt():
    prompt = """Given a chat history and the latest user question \
                which might reference context in the chat history, formulate a standalone question \
                which can be understood without the chat history. Do NOT answer the question, \
                just reformulate it if needed and otherwise return it as is."""

    retriever_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    return retriever_prompt


class RAG:
    def __init__(self):
        self.llm = ChatOllama(model="llama3", temperature=0)
        self.retriever = get_document_retriever()
        self.llm_prompt = get_llm_prompt()
        self.retriever_prompt = get_retriever_prompt()
        self.output_parser = StrOutputParser()
        self.chat_history = []
        self.contextualize_query_chain = self.retriever_prompt | self.llm | self.output_parser
        self.rag_chain = (
                RunnablePassthrough.assign(
                    context=self.contextualized_question | self.retriever | self.format_docs
                )
                | self.llm_prompt
                | self.llm
                | self.output_parser
        )

    def contextualized_question(self, inp: dict):
        if inp.get("chat_history"):
            return self.contextualize_query_chain
        return inp["input"]

    def format_docs(self, documents):
        return "\n\n".join(doc.page_content for doc in documents)

    def enrich_chat_history_human(self, inp):
        self.chat_history.append(HumanMessage(content=inp))

    def enrich_chat_history_ai(self, inp):
        self.chat_history.append(AIMessage(content=inp))

    def generate_response(self, query):
        response = self.rag_chain.invoke({"input": query, "chat_history": self.chat_history})

        self.enrich_chat_history_human(query)
        self.enrich_chat_history_ai(response)

        return response

    def generate_response_streamlit(self, query, chat_history):
        response = self.rag_chain.stream({"input": query, "chat_history": chat_history})
        return response


if __name__ == '__main__':
    rag = RAG()

    disease = input("Enter disease: ")
    user_disease = "I am suffering from " + str(disease)
    rag.enrich_chat_history_human(user_disease)

    while True:
        input_query = input("\nEnter your question: ")
        if input_query.lower() == "exit":
            break
        print("\nLLM Response:")
        print(rag.generate_response(input_query))
