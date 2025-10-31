from .qdrant_client import get_qdrant_client, get_collection_name
from .document_exist import check_document_id_already_exists

def delete_document_from_qdrant(user_id, doc_id, embedding_model):
    try:
        client = get_qdrant_client()
        collection_name = get_collection_name(user_id, embedding_model)
        collections = client.get_collections()
        
        document_exist, status = check_document_id_already_exists(user_id, doc_id, embedding_model)
        if not status:
            print(document_exist)
            return document_exist, False
        
        search_results = client.scroll(
            collection_name=collection_name,
            scroll_filter={
                "must": [
                    {"key": "user_id", "match": {"value": str(user_id)}},
                    {"key": "doc_id", "match": {"value": str(doc_id)}}
                ]
            },
            limit=10000,
            with_payload=False
        )
        if not search_results[0]:
            print(f"📭 No chunks found for document {doc_id}")
            return f"No chunks found for document {doc_id}", False
        point_ids = [point.id for point in search_results[0]]
        client.delete(
            collection_name=collection_name,
            points_selector=point_ids
        )
        print(f"✅ Deleted {len(point_ids)} chunks for document {doc_id}")
        return f"Deleted {len(point_ids)} chunks for document {doc_id}", True
    except Exception as e:
        print(f"❌ Error deleting document {doc_id}: {str(e)}")
        return f"Error deleting document {doc_id}: {str(e)}", False


def delete_collection_from_qdrant(user_id, embedding_model):
    """
    Delete a Qdrant collection for a specific user and embedding model.
    
    Args:
        user_id (str): The user ID
        embedding_model (str): The embedding model name
        
    Returns:
        tuple: (message, success_status)
    """
    try:
        client = get_qdrant_client()
        collection_name = get_collection_name(user_id, embedding_model)
        
        # Check if collection exists before attempting to delete
        try:
            collections = client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if collection_name not in collection_names:
                print(f"📭 Collection {collection_name} does not exist")
                return f"Collection {collection_name} does not exist", True  # Return True as it's not an error
            
            # Delete the collection
            client.delete_collection(collection_name)
            print(f"✅ Successfully deleted collection {collection_name}")
            return f"Successfully deleted collection {collection_name}", True
            
        except Exception as collection_error:
            print(f"❌ Error accessing collections: {str(collection_error)}")
            return f"Error accessing collections: {str(collection_error)}", False
            
    except Exception as e:
        error_msg = f"Error deleting collection for user {user_id} with embedding model {embedding_model}: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg, False
    