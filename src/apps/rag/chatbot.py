import os

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
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_SERVER_URL = "http://ollama:11434"
EMBEDDING_MODEL = "nomic-embed-text"
# EMBEDDING_MODEL="mxbai-embed-large"

DOC_DIR = os.path.join(os.getcwd(), 'src/docs')


from langchain_core.prompts import PromptTemplate

def rag_prompt_mistral_build():

    prompt_template = """
    <s> [INST] Vous êtes un assistant pour les tâches de réponse aux questions. Utilisez les éléments suivants du contexte récupéré pour répondre à la question. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas. Utilisez trois phrases maximum et soyez concis. [/INST] </s> 

    [INST] Question: {question} 
    
    Context: {context} 
    
    Answer: [/INST]
    """

    # Create a PromptTemplate instance with the defined template and input variables
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    # logger.info(f"LANGCHAIN_API_KEY={os.getenv('LANGCHAIN_API_KEY')}")
    # prompt = hub.pull("rlm/rag-prompt-mistral")
    return prompt


def load_docs():
    logger.info("Loading documents...")
    loader = DirectoryLoader(DOC_DIR, glob="**/*.md", show_progress=True, loader_cls=TextLoader)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents.")
    return docs


def split_docs(docs):
    logger.info("Splitting documents...")
    headers_to_split_on = [
        # ("#", "Header1"),
        # ("##", "Header2"),
        # ("###", "Header3"),
        ("####", "Header4"),
        # ("#####", "Header5"),
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


def init_connection():
    docs = load_docs()
    docs = split_docs(docs)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_SERVER_URL)

    logger.info("Initializing vector database...")
    vector_db = Milvus.from_documents(
        docs,
        embeddings,
        connection_args={"host": "milvus", "port": "19530"},
        collection_name="syntheses_conventions_collectives",
        drop_old=True,  # Drop the old Milvus collection if it exists
    )
    logger.info(f"Vector DB initialized and Data imported.")


    logger.info("Initializing vector database...")
    model="mistral"
    llm = Ollama(model=model,  # llama2-uncensored
                 verbose=True,
                 base_url=OLLAMA_SERVER_URL)
    logger.info(f"Connected to Ollama and {model} loaded")

    return vector_db, llm


def main(vector_db, llm):
    st.title("RAG Chatbot 🤖")
    st.write("Posez une question sur les conventions:")
    # st.write(DOC_DIR)

    with st.chat_message('assistant'):
        st.markdown("Hello 👋🏿, Que voulez vous savoir?")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Enter your question here:"):

        # Display user message in chat message container
        st.chat_message("user").markdown(query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})


        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # retrieved_docs = retriever.invoke(query)
        # logger.info(f"Retrieved {len(retrieved_docs)} documents.")


        with st.spinner("Processing..."):
            retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
            prompt = rag_prompt_mistral_build()
            rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
            )
            response_text = rag_chain.invoke(query)

            with st.chat_message("assistant"):
                st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    vector_db, llm = init_connection()
    main(vector_db, llm)
