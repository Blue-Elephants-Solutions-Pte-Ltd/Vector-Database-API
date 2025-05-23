from langchain_chroma import Chroma
import logging
from llm_provider import get_embeddings_model

logger = logging.getLogger(__name__)

def delete_vectors_from_db(user_id, document_id_list, vector_store_path, llm_provider, api_key, embedding_model):
    
    try:
    
        embeddings = get_embeddings_model(llm_provider, embedding_model, api_key)
    
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=vector_store_path
        )

        metadata_filter = {
            "$and": [
                {"user_id": user_id}, 
                {"doc_id": {"$in": document_id_list}}
                ]
            }

        # Access the underlying Chroma collection directly
        matching_docs = vectorstore._collection.get(where=metadata_filter)

        # Extract the IDs of the matching documents
        doc_ids_to_delete = matching_docs["ids"]  # Access the 'ids' key from the returned dictionary
        # total_vectors_deleted = len(doc_ids_to_delete)

        vectorstore.delete(ids=doc_ids_to_delete)
        print("embeddings deleted successfully from vectorstore!!")
        status=f"Document id {document_id_list} successfully deleted from vector store"
    
    except Exception as e:
        logger.error(f"Error in delete_vectors_from_db: {str(e)}")
        status=f"Error in delete_vectors_from_db for document id {document_id_list}: {str(e)}"
    
    return status


def delete_all_vectors_from_db(user_id, vector_store_path, llm_provider, api_key, embedding_model):
    
    try:
    
        embeddings = get_embeddings_model(llm_provider, embedding_model, api_key)
    
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=vector_store_path
        )

        # Simplified where clause since we only have one condition
        metadata_filter = {"user_id": user_id}

        # Delete all documents matching the filter
        vectorstore._collection.delete(where=metadata_filter)
        print("embeddings deleted successfully from vectorstore!!")
        status=f"All documents successfully deleted from vector store"
    
    except Exception as e:
        logger.error(f"Error in delete_all_vectors_from_db: {str(e)}")
        status=f"Error in delete_all_vectors_from_db: {str(e)}"
    
    return status
    
    