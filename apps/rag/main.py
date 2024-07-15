import logging
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_milvus.vectorstores import Milvus
from langchain_community.llms import Ollama
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import hub

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_SERVER_URL = "http://localhost:7869"
EMBEDDING_MODEL = "nomic-embed-text"
# EMBEDDING_MODEL="mxbai-embed-large"

def main():
    logger.info("Starting the document loading process...")
    loader = DirectoryLoader('../embedding/docs', glob="**/*.md", show_progress=True, loader_cls=TextLoader)
    docs = loader.load()

    logger.info("Loaded documents: %d", len(docs))

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

    logger.info("Splitting documents by headers...")

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, add_start_index=True)
    docs = text_splitter.split_documents(docs)

    logger.info("Splitting documents by characters...")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_SERVER_URL)
    query_result = embeddings.embed_query(str(docs[0]))

    logger.info("Embedding query results: %s", query_result[:5])

    MILVUS_CLUSTER_URL = "http://localhost:19530"
    vector_db = Milvus.from_documents(
        docs,
        embeddings,
        connection_args={"uri": MILVUS_CLUSTER_URL},
        collection_name="syntheses_conventions_collectives",
        drop_old=True,  # Drop the old Milvus collection if it exists
    )

    logger.info("Created Milvus vector store")

    prompt = hub.pull("rlm/rag-prompt-mistral")
    llm = Ollama(model="mistral",  # llama2-uncensored
                 verbose=True,
                 base_url=OLLAMA_SERVER_URL)

    logger.info(f"Loaded LLM model {llm.model}")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    retrieved_docs = retriever.invoke("De quel convention parle t'on?")

    logger.info("Retrieved %d documents", len(retrieved_docs))

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    question = "De quel convention parle t'on? "
    logger.info("Asking question: %s", question)

    logger.info("Assistant response: %s", question)
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    main()
