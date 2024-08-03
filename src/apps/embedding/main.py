import os
# Load docs
# from langchain.document_loaders import WebBaseLoader
# # loader = WebBaseLoader("https://www.smartdatapay.com/conventions/KALICONT000005635624.html")
# loader = WebBaseLoader("https://travail-emploi.gouv.fr/droit-du-travail/les-absences-pour-maladie-et-conges-pour-evenements-familiaux/article/les-conges-pour-evenements-familiaux-et-le-conge-de-deuil")
# docs = loader.load()

# print(docs)

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

DOC_DIR = os.path.join(os.getcwd() ,'src/docs')

loader = DirectoryLoader(DOC_DIR, glob="**/*.md", show_progress=True, loader_cls=TextLoader)
docs = loader.load()

# print(docs[:5])


# embedding

from langchain_community.embeddings import OllamaEmbeddings

EMBEDDING_MODEL="nomic-embed-text"
OLLAMA_SERVER_URL="http://localhost:11434"
# EMBEDDING_MODEL="mxbai-embed-large"

embeddings = (
    OllamaEmbeddings(model=EMBEDDING_MODEL, base_url = OLLAMA_SERVER_URL )
)

# query_result = embeddings.embed_query(str(docs[0]))

# Split

from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    # ("#", "Header1"),
    # ("##", "Header2"),
    # ("###", "Header3"),
    ("####", "Header4"),
    # ("#####", "Header5"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)
md_header_splits = []
for docu in docs:
    md_header_splits.extend(markdown_splitter.split_text(docu.page_content))


# from langchain_experimental.text_splitter import SemanticChunker
# text_splitter = SemanticChunker(embeddings)
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)
# split_documents = text_splitter.create_documents([format_docs(docs)])

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    length_function=len,
    add_start_index=True,
    keep_separator=True
)
split_documents = text_splitter.split_documents(docs)
# split_documents = text_splitter.create_documents([doc.page_content for doc in docs])



# print(md_header_splits[:5])
# print(docs[:5])
# # print(docs[-1:5])
#
# exit(0)
#




from langchain_milvus.vectorstores import Milvus

# The easiest way is to use Milvus Lite where everything is stored in a local file.
# If you have a Milvus server you can use the server URI such as "http://localhost:19530".
# URI = "./milvus_demo.db"
URI = "http://localhost:19530"


vector_db = Milvus.from_documents(
    md_header_splits,
    embeddings,
    connection_args={"uri": URI},
    collection_name="syntheses_ccn",
    drop_old=True,  # Drop the old Milvus collection if it exists
)

vector_db = Milvus.from_documents(
    split_documents,
    embeddings,
    connection_args={"uri": URI},
    collection_name="syntheses_ccn_splits",
    drop_old=True,  # Drop the old Milvus collection if it exists
)

# query = "Les congés pour événements familiaux et le congé de deuil"
# docs = vector_db.similarity_search(query)
#
# print(docs[:5])