from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
import threading
import os
import dotenv
dotenv.load_dotenv()

def get_qdrant_client():
    """Get Qdrant client instance using environment variables.

    Expects:
    - QDRANT_URL (e.g., http://localhost:6333 or http://qdrant:6333)
    - QDRANT_API_KEY (optional; if set, used for auth)
    """
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    if not api_key:
        print("⚠️ QDRANT_API_KEY not set. If your Qdrant instance enforces auth, requests will fail with 401.")
    return QdrantClient(url=url, api_key=api_key, timeout=52.0, check_compatibility=False)


# Collection naming configuration
QDRANT_COLLECTION_PREFIX =  "ai_knowledge_assistant_"

def get_collection_name(user_id, embedding_model):
    """Generate collection name for a user and embedding model combination

    Format: {PREFIX}{user_id}_{embedding_model}, with model sanitized for '-', '.', '/'.
    """
    safe_embedding_model = (
        embedding_model
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )
    return f"{QDRANT_COLLECTION_PREFIX}{user_id}_{safe_embedding_model}"

def get_embedding_dimension(embeddings):
    """Get embedding dimension dynamically from the embeddings model"""
    try:
        test_text = "test"
        test_embedding = embeddings.embed_query(test_text)
        return len(test_embedding)
    except Exception as e:
        print(f"❌ Error getting embedding dimension: {str(e)}")
        return None

# Thread lock for collection creation to prevent race conditions
_collection_creation_lock = threading.Lock()

def ensure_collection_exists(client, collection_name, embeddings):
    """Ensure Qdrant collection exists with proper configuration"""
    try:
        # Use thread lock to prevent multiple threads from creating the same collection
        with _collection_creation_lock:
            collections = client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if collection_name not in collection_names:
                vector_size = get_embedding_dimension(embeddings)
                print(f"🔧 Creating new collection: {collection_name}")
                print(f"   Embedding dimension detected: {vector_size}")
                
                try:
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            "size": vector_size,
                            "distance": "Cosine"
                        }
                    )
                    print(f"✅ Created Qdrant collection: {collection_name} with dimension: {vector_size}")
                except UnexpectedResponse as e:
                    if "already exists" in str(e):
                        print(f"✅ Collection {collection_name} was created by another thread")
                    else:
                        raise e
            else:
                try:
                    collection_info = client.get_collection(collection_name)
                    existing_vector_size = getattr(collection_info.config.params.vectors, 'size', None)
                    if existing_vector_size is None:
                        print(f"⚠️ Collection {collection_name} exists but has invalid configuration. Recreating...")
                        client.delete_collection(collection_name)
                        vector_size = get_embedding_dimension(embeddings)
                        client.create_collection(
                            collection_name=collection_name,
                            vectors_config={
                                "size": vector_size,
                                "distance": "Cosine"
                            }
                        )
                        print(f"✅ Recreated Qdrant collection: {collection_name} with dimension: {vector_size}")
                    else:
                        print(f"✅ Using existing Qdrant collection: {collection_name}")
                        print(f"   Existing vector size: {existing_vector_size}")
                        expected_vector_size = get_embedding_dimension(embeddings)
                        if existing_vector_size != expected_vector_size:
                            print(f"⚠️  WARNING: Dimension mismatch detected!")
                            print(f"   Collection name suggests: {collection_name}")
                            print(f"   Existing vector size: {existing_vector_size}")
                            print(f"   Expected vector size: {expected_vector_size}")
                            print(f"   This may indicate a collection naming issue")
                except Exception as config_error:
                    print(f"⚠️ Error checking collection {collection_name} configuration: {config_error}. Recreating...")
                    try:
                        client.delete_collection(collection_name)
                    except:
                        pass
                    vector_size = get_embedding_dimension(embeddings)
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            "size": vector_size,
                            "distance": "Cosine"
                        }
                    )
                    print(f"✅ Recreated Qdrant collection: {collection_name} with dimension: {vector_size}")
    except Exception as e:
        print(f"❌ Error ensuring collection exists: {str(e)}")
        # Don't raise the error if it's just a collection already exists error
        if "already exists" not in str(e):
            raise e
