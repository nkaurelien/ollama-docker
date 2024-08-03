
from langchain_community.embeddings import OllamaEmbeddings
import json
from pymilvus import MilvusClient

CLUSTER_ENDPOINT="http://localhost:19530"
# 1. Set up a Milvus client
client = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    # token=TOKEN
)

EMBEDDING_MODEL="nomic-embed-text"
OLLAMA_SERVER_URL="http://localhost:11434"

embeddings = (
    OllamaEmbeddings(model=EMBEDDING_MODEL, base_url = OLLAMA_SERVER_URL )
)

query = "de quel convention parle t-on?"

query_embedding = embeddings.embed_query(query)

print(query_embedding[:5])

from langchain_milvus.vectorstores import Milvus

# The easiest way is to use Milvus Lite where everything is stored in a local file.
# If you have a Milvus server you can use the server URI such as "http://localhost:19530".
# URI = "./milvus_demo.db"
URI = "http://localhost:19530"


# Single vector search
res = client.search(
    collection_name="syntheses_conventions_collectives",
    # Replace with your query vector
    # data=[[0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]],
    data=[query_embedding],
    limit=5, # Max. number of search results to return
    # search_params={"metric_type": "IP", "params": {}} # Search parameters
)

# Convert the output to a formatted JSON string
result = json.dumps(res, indent=4)
print(result)
