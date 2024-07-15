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

OLLAMA_SERVER_URL = "http://localhost:7869"
EMBEDDING_MODEL = "nomic-embed-text"
# EMBEDDING_MODEL="mxbai-embed-large"

def main():
    loader = DirectoryLoader('../embedding/docs', glob="**/*.md", show_progress=True, loader_cls=TextLoader)
    docs = loader.load()

    # print(docs[:5])



    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=True
    )
    md_header_splits = []
    for docu in docs:
        md_header_splits.extend(markdown_splitter.split_text(docu.page_content))

    # Split
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    docs = text_splitter.split_documents(docs)

    # embedding

    embeddings = (
        OllamaEmbeddings(model=EMBEDDING_MODEL, base_url = OLLAMA_SERVER_URL )
    )

    query_result = embeddings.embed_query(str(docs[0]))

    print(query_result[:5])


    # The easiest way is to use Milvus Lite where everything is stored in a local file.
    # If you have a Milvus server you can use the server URI such as "http://localhost:19530".
    # URI = "./milvus_demo.db"
    MILVUS_CLUSTER_URL = "http://localhost:19530"

    vector_db = Milvus.from_documents(
        docs,
        embeddings,
        connection_args={"uri": MILVUS_CLUSTER_URL},
        collection_name="syntheses_conventions_collectives",
        drop_old=True,  # Drop the old Milvus collection if it exists
    )

    # query = "Les congés pour événements familiaux et le congé de deuil"
    # docs = vector_db.similarity_search(query)
    #
    # print(docs[:5])

    # RAG prompt
    from langchain import hub
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = Ollama(model="mystal", # llama2-uncensored
                 verbose=True,
                 base_url=OLLAMA_SERVER_URL)

    print(f"Loaded LLM model {llm.model}")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vector_db.as_retriever()


    # retrieved_docs = retriever.invoke("De quel convention parle t'on?")
    #
    # len(retrieved_docs)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # Ask a question

    question = f"De quel convention parle t'on? "
    # rag_chain.invoke(question)

    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    main()