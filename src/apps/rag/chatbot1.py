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
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# import multiprocessing as mp
# mp.set_start_method("spawn", force=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_SERVER_URL = "http://ollama:11434"
EMBEDDING_MODEL = "nomic-embed-text"
# EMBEDDING_MODEL="mxbai-embed-large"

DOC_DIR = os.path.join(os.getcwd() ,'src/docs')

from langchain_core.prompts import PromptTemplate
def rag_prompt_mistral_build():
    # Define the prompt template for generating AI responses
    # PROMPT_TEMPLATE = """
    # Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
    # Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # <context>
    # {context}
    # </context>
    # 
    # <question>
    # {question}
    # </question>
    # 
    # The response should be specific and use statistics or numbers when possible.
    # 
    # Assistant:"""

    PROMPT_TEMPLATE = """
    <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. [/INST] </s> 

    [INST] Question: {question} 
    
    Context: {context} 
    
    Answer: [/INST]
    """

    # Create a PromptTemplate instance with the defined template and input variables
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    # logger.info(f"LANGCHAIN_API_KEY={os.getenv('LANGCHAIN_API_KEY')}")
    # prompt = hub.pull("rlm/rag-prompt-mistral")
    return prompt



llm = Ollama(model="mistral",  # llama2-uncensored
             verbose=True,
             base_url=OLLAMA_SERVER_URL)
logger.info(f"Connected to Ollama.")

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
def init_vector_db():
    docs = load_docs()
    docs = split_docs(docs)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_SERVER_URL)

    milvus_cluster_url = "http://milvus:19530"

    logger.info("Initializing vector database...")
    vector_db = Milvus.from_documents(
        docs,
        embeddings,
        connection_args={"host": "milvus", "port": "19530"},
        collection_name="syntheses_conventions_collectives",
        drop_old=True,  # Drop the old Milvus collection if it exists
    )

    return vector_db


def main(vector_db):
    st.title("RAG Chatbot")
    st.write("Ask a question about the conventions:")
    # st.write(DOC_DIR)

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

        with st.spinner("Processing..."):


            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
            retrieved_docs = retriever.invoke(query)

            logger.info(f"Retrieved {len(retrieved_docs)} documents.")

            prompt = rag_prompt_mistral_build()

            rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
            )

            # response_area = st.empty()

            response_text = ""
            with st.chat_message("assistant"):
            # Add user message to chat history
                for chunk in rag_chain.stream(query):
                    response_text += chunk
                    # response_area.text(response_text)
                    st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    vector_db = init_vector_db()
    main(vector_db)
