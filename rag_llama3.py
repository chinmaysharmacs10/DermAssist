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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"

local_llm = "llama3"
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

urls = [
    "https://www.aad.org/public/diseases/acne/diy/adult-acne-treatment",
]


def rag(query):
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

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
    )

    llm = ChatOllama(model=local_llm, temperature=0)

    # Post-processing
    def format_docs(documents):
        return "\n\n".join(doc.page_content for doc in documents)

    # Chain
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    generation = rag_chain.invoke(query)
    return generation


def test_llama(query):
    llm = ChatOllama(model=local_llm)
    prompt = ChatPromptTemplate.from_template("{query}")
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": query})
    return response


if __name__ == '__main__':
    input_query = "what non-prescription treatments can I use to treat?"
    print(rag(input_query))


