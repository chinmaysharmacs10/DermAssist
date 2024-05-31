import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

urls = [
    "https://www.aad.org/public/diseases/acne/diy/adult-acne-treatment",
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
            Question: {question} 
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

    def format_docs(self, documents):
        return "\n\n".join(doc.page_content for doc in documents)

    def generate_response(self, query):
        rag_chain = (
                {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
        )
        response = rag_chain.invoke(query)

        return response


if __name__ == '__main__':
    rag = RAG()
    while True:
        input_query = input("Enter your question: ")
        if input_query == "stop":
            break
        print("\nLLM Response:")
        print(rag.generate_response(input_query), "\n")
