from .qdrant_client import get_qdrant_client, get_collection_name, ensure_collection_exists

def get_user_collections(client, user_id):
    try:
        prefix = f"ai_knowledge_assistant_{user_id}_"
        collections = client.get_collections()
        return [col.name for col in collections.collections if col.name.startswith(prefix)]
    except Exception as e:
        print(f"❌ Error getting user collections: {str(e)}")
        return []

def delete_user_collections(client, user_id):
    try:
        user_collections = get_user_collections(client, user_id)
        deleted_count = 0
        for collection_name in user_collections:
            try:
                client.delete_collection(collection_name)
                print(f"✅ Deleted collection: {collection_name}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Error deleting collection {collection_name}: {str(e)}")
        print(f"✅ Deleted {deleted_count} collections for user {user_id}")
        return deleted_count
    except Exception as e:
        print(f"❌ Error deleting user collections: {str(e)}")
        return 0

def delete_user_embedding_collection(client, user_id, embedding_model):
    try:
        collection_name = get_collection_name(user_id, embedding_model)
        client.delete_collection(collection_name)
        print(f"✅ Deleted collection: {collection_name}")
        return True
    except Exception as e:
        print(f"❌ Error deleting collection {collection_name}: {str(e)}")
        return False

def handle_embedding_model_change(user_id, old_embedding_model, new_embedding_model):
    try:
        client = get_qdrant_client()
        old_collection = get_collection_name(user_id, old_embedding_model)
        new_collection = get_collection_name(user_id, new_embedding_model)
        delete_user_embedding_collection(client, user_id, old_embedding_model)
        ensure_collection_exists(client, new_collection, None)  # Embeddings should be passed as needed
        return True
    except Exception as e:
        print(f"❌ Error handling embedding model change: {str(e)}")
        return False

def get_user_embedding_collections_info(user_id):
    try:
        client = get_qdrant_client()
        collections = get_user_collections(client, user_id)
        info = []
        for collection_name in collections:
            try:
                collection_info = client.get_collection(collection_name)
                points_count = collection_info.points_count
                vectors_count = collection_info.vectors_count
                vector_size = collection_info.config.params.vectors.size
                info.append({
                    "collection_name": collection_name,
                    "points_count": points_count,
                    "vectors_count": vectors_count,
                    "vector_size": vector_size
                })
            except Exception as e:
                info.append({"collection_name": collection_name, "error": str(e)})
        return info
    except Exception as e:
        print(f"❌ Error getting user collections info: {str(e)}")
        return []

def get_collection_stats(client, collection_name):
    try:
        collection_info = client.get_collection(collection_name)
        return {
            "points_count": collection_info.points_count,
            "vectors_count": collection_info.vectors_count,
            "vector_size": collection_info.config.params.vectors.size
        }
    except Exception as e:
        print(f"❌ Error getting collection stats: {str(e)}")
        return {}
    