import streamlit as st
import logging
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_milvus.vectorstores import Milvus
from langchain_community.llms import Ollama
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_SERVER_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
# EMBEDDING_MODEL="mxbai-embed-large"

def load_docs():
    doc_dir = os.path.join(os.getcwd(), 'src/docs')

    logger.info("Loading documents...")
    loader = DirectoryLoader(doc_dir, glob="**/*.md", show_progress=True, loader_cls=TextLoader)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents.")
    return docs

def split_docs(docs):
    logger.info("Splitting documents...")
    headers_to_split_on = [
        ("#", "Header1"),
        ("##", "Header2"),
        ("###", "Header3"),
        ("####", "Header4"),
        ("#####", "Header5"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=True
    )
    md_header_splits = []
    for docu in docs:
        md_header_splits.extend(markdown_splitter.split_text(docu.page_content))

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, add_start_index=True)
    split_docs = text_splitter.split_documents(docs)
    logger.info(f"Split into {len(split_docs)} chunks.")
    return split_docs

def main():
    st.title("RAG with Streamlit")
    st.write("Ask a question about the conventions:")

    query = st.text_input("Enter your question here:")

    if st.button("Submit"):
        with st.spinner("Processing..."):
            docs = load_docs()
            docs = split_docs(docs)

            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_SERVER_URL)

            MILVUS_CLUSTER_URL = "http://localhost:19530"

            logger.info("Initializing vector database...")
            vector_db = Milvus.from_documents(
                docs,
                embeddings,
                connection_args={"uri": MILVUS_CLUSTER_URL},
                collection_name="syntheses_conventions_collectives",
                drop_old=True,  # Drop the old Milvus collection if it exists
            )

            from langchain import hub
            prompt = hub.pull("rlm/rag-prompt-mistral")

            llm = Ollama(model="mistral", # llama2-uncensored
                         verbose=True,
                         base_url=OLLAMA_SERVER_URL)

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
            retrieved_docs = retriever.invoke(query)

            logger.info(f"Retrieved {len(retrieved_docs)} documents.")

            rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
            )

            response_area = st.empty()

            response_text = ""
            for chunk in rag_chain.stream(query):
                response_text += chunk
                response_area.text(response_text)

if __name__ == "__main__":
    main()
