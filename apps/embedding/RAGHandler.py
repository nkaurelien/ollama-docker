from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility, Index
import ollama
import os

from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility, Index
import ollama
import os


class RAGHandler:
    def __init__(self, milvus_host="127.0.0.1", milvus_port="19530"):
        # Connect to Milvus
        connections.connect("default", host=milvus_host, port=milvus_port)
        print(f"Connected to Milvus at {milvus_host}:{milvus_port}")


    def create_collection(self, collection_name, embedding_dim=768, max_length=512):
        """
        Creates a new collection in Milvus with the specified schema.

        Args:
        - collection_name (str): The name of the collection to create or access.
        - embedding_dim (int, optional): The dimension of the embedding vectors. Default is 768.
        - max_length (int, optional): The maximum length for the document text field. Default is 512.

        Returns:
        - Collection: A Milvus collection object representing the created or accessed collection.
        """

        # Define the fields for the collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=max_length)
        ]
        schema = CollectionSchema(fields, f"Document embeddings with Ollama - {collection_name}")

        # Check if the collection already exists
        if not utility.has_collection(collection_name):
            # Create the collection if it does not exist
            collection = Collection(collection_name, schema)
            print(f"Collection '{collection_name}' created.")
        else:
            # Access the existing collection
            collection = Collection(collection_name)
            print(f"Collection '{collection_name}' already exists.")

        return collection

    def insert_data(self, collection_name, documents):
        """
        Inserts data into the specified Milvus collection.

        Args:
        - collection_name (str): The name of the collection to insert data into.
        - documents (list of str): The documents to be inserted into the collection.

        Returns:
        - None
        """
        collection = Collection(collection_name)
        ids = [i for i in range(len(documents))]
        embeddings = []
        for doc in documents:
            embedding = self.generate_embeddings(doc)
            if embedding:
                embeddings.append(embedding)

        if not embeddings:
            print(f"Error: No embeddings generated for collection '{collection_name}'. Data not inserted.")
            return

        collection.insert([ids, embeddings, documents])
        print(f"Inserted {len(documents)} documents into the collection '{collection_name}'.")

    def create_index(self, collection_name):
        """
        Creates an index for the specified collection in Milvus.

        Args:
        - collection_name (str): The name of the collection to create an index for.

        Returns:
        - None
        """
        collection = Collection(collection_name)
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"Index created for collection '{collection_name}'.")

    def load_collections(self, collection_names):
        """
        Loads the specified collections into memory in Milvus.

        Args:
        - collection_names (list of str): A list of collection names to load.

        Returns:
        - None
        """
        for name in collection_names:
            collection = Collection(name)
            collection.load()
            print(f"Collection '{name}' loaded into memory.")

    def generate_embeddings(self, text):
        """
        Generates embeddings for the given text using the specified model.

        Args:
        - text (str): The text to generate embeddings for.

        Returns:
        - list: The generated embeddings or None if there was an error.
        """
        try:
            response = ollama.embeddings(model="nomic-embed-text", prompt=text)
            embedding = response.get("embedding")
            if not embedding:
                print(f"Warning: No 'embedding' in response for text: {text}")
            return embedding
        except Exception as e:
            print(f"Error generating embeddings for text: {text} - {e}")
            return None

    def generate_response(self, prompt, context):
        """
        Generates a response based on the given prompt and context using the specified model.

        Args:
        - prompt (str): The prompt to generate a response for.
        - context (str): The context to use for generating the response.

        Returns:
        - dict: The generated response or None if there was an error.
        """
        try:
            response = ollama.generate(model="mistral",
                                       prompt=f"Using this data: {context}. Respond to this prompt: {prompt}")
            return response
        except Exception as e:
            print(f"Error generating response for prompt: {prompt} - {e}")
            return None

    def query_and_generate_response(self, query_text, collection_name):
        """
        Queries the specified collection for similar documents and generates a response.

        Args:
        - query_text (str): The text to query the collection with.
        - collection_name (str): The name of the collection to query.

        Returns:
        - dict: The generated response or None if there was an error.
        """
        query_embedding = self.generate_embeddings(query_text)
        if query_embedding is None:
            print(f"Error: Failed to generate embedding for query: {query_text}")
            return None

        collection = Collection(collection_name)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        try:
            results = collection.search([query_embedding], "embedding", search_params, limit=3,
                                        output_fields=["document"])
            context_documents = [result.entity.get("document") for result in results[0]]
            context = " ".join(context_documents)
            response = self.generate_response(query_text, context)
            return response
        except Exception as e:
            print(f"Error querying collection '{collection_name}' and generating response: {e}")
            return None

    def extract_response_text(self, response):
        """
        Extracts the text response from the verbose response object.

        Args:
        - response (dict): The response object containing the generated text.

        Returns:
        - str: The extracted text response.
        """
        if isinstance(response, dict) and "response" in response:
            return response["response"]
        return "No response found"

    def list_collections(self, detailed):
        """
        Lists all collections in Milvus with information about their state.

        Args:
        - detailed (boolean): simple or detailed answer.

        Returns:
        - list: A list of dictionaries containing collection names and their state information.
        """
        collection_info_list = []
        try:
            collections = utility.list_collections()
            for collection_name in collections:
                collection = Collection(collection_name)
                info = collection.describe()
                num_entities = collection.num_entities
                if (detailed):
                    collection_info_list.append({
                        "collection_name": collection_name,
                        "description": info,
                        "num_entities": num_entities
                    })
                else:
                    collection_info_list.append({
                        "collection_name": collection_name,
                        "num_entities": num_entities
                    })

        except Exception as e:
            print(f"Error listing collections: {e}")
        return collection_info_list

    def delete_collection(self, collection_name):
        """
        Deletes a specified collection from Milvus.

        Args:
        - collection_name (str): The name of the collection to delete.

        Returns:
        - str: A message indicating the result of the deletion.
        """
        try:
            collection = Collection(collection_name)
            collection.drop()
            return f"Collection '{collection_name}' has been deleted."
        except Exception as e:
            return f"Error deleting collection '{collection_name}': {e}"

    def expand_collection(self, collection_name, new_documents):
        """
        Expands a collection by adding multiple new documents.

        Args:
        - collection_name (str): The name of the collection to expand.
        - new_documents (list of str): The new documents to add to the collection.

        Returns:
        - str: A message indicating the result of the operation.
        """
        collection = Collection(collection_name)
        current_count = collection.num_entities

        ids = [current_count + i for i in range(len(new_documents))]
        embeddings = []

        for doc in new_documents:
            embedding = self.generate_embeddings(doc)
            if embedding:
                embeddings.append(embedding)

        if not embeddings:
            return f"Error: No embeddings generated for new documents in collection '{collection_name}'. Data not inserted."

        collection.insert([ids, embeddings, new_documents])
        return f"Inserted {len(new_documents)} new documents into the collection '{collection_name}'."


# Example usage:
if __name__ == "__main__":
    handler = RAGHandler()

    # Create collections
    collection_name1 = "example_collection1"
    collection_name2 = "example_collection2"
    handler.create_collection(collection_name1)
    handler.create_collection(collection_name2)

    # Insert some documents
    documents1 = [
        "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
        "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
        "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall"
    ]
    documents2 = [
        "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
        "Llamas are vegetarians and have very efficient digestive systems",
        "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old"
    ]
    handler.insert_data(collection_name1, documents1)
    handler.insert_data(collection_name2, documents2)

    # Create an index and load the collections
    handler.create_index(collection_name1)
    handler.create_index(collection_name2)
    handler.load_collections([collection_name1, collection_name2])

    # Query and generate a response
    query_text = "What animals are llamas related to?"
    response1 = handler.query_and_generate_response(query_text, collection_name1)
    response2 = handler.query_and_generate_response(query_text, collection_name2)
    # print(response1)
    # print(response2)
    clean_response1 = handler.extract_response_text(response1)
    print(clean_response1)
    clean_response2 = handler.extract_response_text(response2)
    print(clean_response2)

    # liste les collections présentes
    collections_info = handler.list_collections(detailed=False)
    for info in collections_info:
        print(info)

    # effacement d'une collection
    delete_message = handler.delete_collection(collection_name1)
    print(delete_message)

    # Expanding a collection
    new_documents = [
        "Llamas are very social animals and live with other llamas as a herd.",
        "Their wool is very soft and lanolin-free.",
        "Llamas can learn simple tasks after a few repetitions."
    ]
    expand_message = handler.expand_collection(collection_name2, new_documents)
    print(expand_message)
    query_text = "What llamas wool is reknown for not having?"  # Query to check extension
    response2 = handler.query_and_generate_response(query_text, collection_name2)
    clean_response2 = handler.extract_response_text(response2)
    print(clean_response2)