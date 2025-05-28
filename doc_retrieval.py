from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_chroma import Chroma
from llm_provider import get_embeddings_model, get_chat_model
import logging
logger = logging.getLogger(__name__)

def create_base_retriever(vectorstore, user_id, document_id_list, score_threshold, k_value):
    """Create a base retriever with configured filtering."""
    retriever = vectorstore.as_retriever()
    retriever.search_type = "similarity_score_threshold"
    
    retriever.search_kwargs = {
        "filter": {
            "$and": [
                {"user_id": user_id},
                {"doc_id": {"$in": document_id_list}}
            ]
        },
        "score_threshold": float(score_threshold),
        "k": int(k_value)
    }

    return retriever

def doc_retriever(query, user_id, document_id_list, score_threshold, k_value, llm_provider, api_key, embedding_model, vectorstore_path, chat_model):
    try:
        if not query:
            raise ValueError("Query cannot be empty")
            
        # Initialize embeddings
        embedding_function = get_embeddings_model(llm_provider, embedding_model, api_key)
        
        llm = get_chat_model(llm_provider, chat_model, api_key)
        
        # Initialize vector store
        vectorstore = Chroma(
            embedding_function=embedding_function,
            persist_directory=vectorstore_path
        )
        
        chunk_count = vectorstore._collection.count()
        k_value = min(k_value, chunk_count)
        
        # Create standard retriever
        standard_retriever = create_base_retriever(
            vectorstore=vectorstore,
            user_id=user_id,
            document_id_list=document_id_list,
            score_threshold=score_threshold,
            k_value=k_value
        )
        
        # Create contextual retriever with lower threshold
        contextual_retriever = create_base_retriever(
            vectorstore=vectorstore,
            user_id=user_id,
            document_id_list=document_id_list,
            score_threshold=0.10,  # Lower threshold for contextual search
            k_value=k_value
        )
        
        # Get standard results
        standard_retriever_docs = standard_retriever.invoke(query)
        logger.info(f'Retrieved {len(standard_retriever_docs)} documents from standard retriever')
        
        # Create and configure contextual compression
        compressor = LLMChainExtractor.from_llm(llm)
        contextual_compression_retriever = ContextualCompressionRetriever(
            base_retriever=contextual_retriever,
            base_compressor=compressor
        )
        
        # Get contextual results
        contextual_retriever_docs = contextual_compression_retriever.invoke(query)
        logger.info(f'Retrieved {len(contextual_retriever_docs)} documents from contextual retriever')
        
        # Combine results
        retrieved_docs = contextual_retriever_docs + standard_retriever_docs
        
        if not retrieved_docs:
            logger.warning("No documents were retrieved for the given query")
            return {
                "status": "success",
                "results": []
            }
        
        return {
            "status": "success",
            "results": retrieved_docs
        }
        
    except ValueError as e:
        logger.error(f"Validation error in doc_retriever: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "results": []
        }
    except Exception as e:
        logger.error(f"Error in doc_retriever: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "results": []
        }