from sentence_transformers import SentenceTransformer
import chromadb

#embedding model loading
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="devops-docs")

def add_document(docs):
    embeddings = embedder.encode(docs).tolist()
    ids = [str(i) for i in range(len(docs))]
    collection.add(
        ids = ids,
        documents = docs,
        embeddings = embeddings
    )

#query the collection chromadb will automatically do it
def query_document(question, k=3):
    q_embedding = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=k
    )
    return results["documents"][0]
    