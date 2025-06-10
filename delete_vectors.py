from langchain_chroma import Chroma
import logging
from llm_provider import get_embeddings_model

logger = logging.getLogger(__name__)

def delete_vectors_from_db(user_id, document_id_list, vector_store_path, llm_provider, api_key, embedding_model):
    
    file_deletion_status = []
    vectorstore = None
    embeddings = None
    try:
        # Initialize embeddings and vectorstore
        embeddings = get_embeddings_model(llm_provider, embedding_model, api_key)
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=vector_store_path
        )
        
        for file_data in document_id_list:
            document_id = file_data.get("document_id")
            
            if document_id is None:
                error_msg = 'Document ID cannot be None'
                logger.warning(f"Invalid document ID: {error_msg}")
                file_deletion_status.append({
                    "user_id": user_id,
                    "document_id": None, 
                    "status": "error", 
                    "message": error_msg
                })
                continue
            
            document_id = str(document_id)
            if not document_id.isdigit() or int(document_id) < 0:
                error_msg = 'Document ID must be a positive integer'
                logger.warning(f"Invalid document ID format: {document_id}")
                file_deletion_status.append({
                    "user_id": user_id,
                    "document_id": document_id, 
                    "status": "error", 
                    "message": error_msg
                })
                continue

            try:
                document_id_list = [document_id]
                metadata_filter = {
                    "$and": [
                        {"user_id": user_id}, 
                        {"doc_id": {"$in": document_id_list}}
                    ]
                }

                matching_docs = vectorstore._collection.get(where=metadata_filter)
                if not matching_docs["ids"]:
                    error_msg = f"No documents found with ID {document_id}"
                    logger.info(error_msg)
                    file_deletion_status.append({
                        "user_id": user_id,
                        "document_id": document_id, 
                        "status": "success", 
                        "message": error_msg
                    })
                    continue

                doc_ids_to_delete = matching_docs["ids"]
                vectorstore.delete(ids=doc_ids_to_delete)
                
                success_msg = f"Document id {document_id} successfully deleted from vector store"
                logger.info(success_msg)
                file_deletion_status.append({
                    "user_id": user_id,
                    "document_id": document_id, 
                    "status": "success", 
                    "message": success_msg
                })
                
            except Exception as e:
                error_msg = f"Error deleting document {document_id}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                file_deletion_status.append({
                    "user_id": user_id,
                    "document_id": document_id, 
                    "status": "success", 
                    "message": error_msg
                })
        
        if not file_deletion_status:
            error_msg = "No documents were processed for deletion"
            logger.warning(error_msg)
            file_deletion_status.append({
                "user_id": user_id,
                "document_id": document_id_list if isinstance(document_id_list, list) else None,
                "status": "success", 
                "message": error_msg
            })
            
    except Exception as e:
        error_msg = f"Error in delete_vectors_from_db: {str(e)}"
        logger.error(error_msg, exc_info=True)
        file_deletion_status.append({
            "user_id": user_id,
            "document_id": document_id_list if isinstance(document_id_list, list) else None,
            "status": "success", 
            "message": error_msg
        })
    
    
    return file_deletion_status


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
    
    