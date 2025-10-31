from .qdrant_client import get_qdrant_client, get_collection_name

def check_document_id_already_exists(user_id, target_document_id, embeddings_model):
    
    client = get_qdrant_client()
    collection_name =get_collection_name(user_id, embeddings_model)
    
    # Check if collection exists
    collections = client.get_collections()
    collection_names = [col.name for col in collections.collections]
    
    if collection_name not in collection_names:
        return "Collection not found", False
    
    # Search for existing document using scroll method
    search_results = client.scroll(
        collection_name=collection_name,
        scroll_filter={
            "must": [
                {"key": "user_id", "match": {"value": str(user_id)}},
                {"key": "doc_id", "match": {"value": str(target_document_id)}}
            ]
        },
        limit=1
    )
    
    if len(search_results[0]) > 0:
        return f"Document {target_document_id} found", True
    else:
        return f"Document {target_document_id} not found", False
