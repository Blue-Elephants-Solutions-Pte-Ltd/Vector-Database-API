# Standard library imports
import os
import logging
import sys
from pathlib import Path
import shutil
# Third-party imports
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from batch_embed import vectorestore_function
from llm_provider import get_embeddings_model
from doc_retrieval import doc_retriever
from utils import format_docs, get_metadata_from_docs
from vector_config.delete_vector_store import delete_document_from_qdrant
from vector_config.delete_vector_store import delete_collection_from_qdrant
from vector_config.qdrant_client import get_collection_name
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Qdrant settings (provided via docker-compose env)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def build_collection_name(user_id: str, embeddings_model: str) -> str:
    # Backwards-compat shim: delegate to centralized helper
    return get_collection_name(user_id, embeddings_model)

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    logger.info("Index route accessed")
    return jsonify({
        "status": "success",
        "message": "Server is running"
    }), 200

@app.route('/add_vectors', methods=['POST'])
def add_vectors():
    try:
        data = request.get_json()
        
        logger.info("===== Add Vectors Request =====")
        logger.info(f"Request data: {data}")
        logger.info("----------------------------------------------------------------")
        
        # Get embeddings and document from request
        user_id = data['user_id']
        doc_id = data['doc_id']
        normalized_doc = data['normalized_doc']
        total_tokens_count = data['total_tokens_count']
        max_token_per_min = data['max_token_per_min']
        llm_provider = data.get('llm_provider')
        api_key = data['api_key']
        embeddings_model = data['embeddings_model']
        file_name = data.get('file_name', 'unknown')
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        split_docs = text_splitter.create_documents(normalized_doc)
        logger.info(f"Number of split documents: {len(split_docs)}")
        
        split_documents_with_metadata = [
            Document(page_content=document.page_content, metadata={"user_id": user_id, "doc_id": doc_id,"file_name": file_name})
            for document in split_docs
        ]
        
        split_docs_length = len(split_documents_with_metadata)
        logger.info(f"Split documents with metadata length: {split_docs_length}")
        
        collection_name = build_collection_name(user_id, embeddings_model)
        # os.makedirs(user_vector_store, exist_ok=True)
        # os.chmod(user_vector_store, 0o777)  # Give full permissions to ensure write access
        
        # Create OpenAIEmbeddings instance
        embedding_function = get_embeddings_model(llm_provider, embeddings_model, api_key)
        logger.info(f"Embedding function created: {embedding_function}")

        status, message = vectorestore_function(
            split_documents_with_metadata,
            collection_name,
            embedding_function,
            max_token_per_min,
            total_tokens_count
        )
        
        if status == "success":
            return jsonify({
                "status": "success",
                "message": message
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": message
            }), 500
    except Exception as e:
        logger.error(f"Error in add_vectors: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route('/retrieve_documents', methods=['POST'])
def retrieve_documents():
    try:
        data = request.get_json()
        logger.info("===== Retrieve Documents Request =====")
        logger.info(f"Request data: {data}")
        logger.info("----------------------------------------------------------------")
        
        # Extract parameters from request
        query = data['query']
        user_id = data.get('user_id')
        document_id_list = data.get('document_id_list')
        score_threshold = data.get('score_threshold', 0.7)  # Default threshold
        k_value = data.get('k_value', 5)  # Default number of results
        llm_provider = data.get('llm_provider', 'openai')
        api_key = data['api_key']
        embedding_model = data['embedding_model']
        chat_model = data['chat_model']

        collection_name = build_collection_name(user_id, embedding_model)
        
        logger.info("Starting document retrieval...")
        # Call the document retriever
        results = doc_retriever(
            query=query,
            user_id=user_id,
            document_id_list=document_id_list,
            score_threshold=score_threshold,
            k_value=k_value,
            llm_provider=llm_provider,
            api_key=api_key,
            embedding_model=embedding_model,
            collection_name=collection_name,
            chat_model=chat_model
        )
        
        retrieved_docs = results['results']
        logger.info(f"Number of retrieved documents: {len(retrieved_docs)}")
        
        try:
            context = format_docs(retrieved_docs)
            context_metadata = get_metadata_from_docs(retrieved_docs)
            logger.info("Documents formatted successfully")
        except Exception as e:
            logger.error(f"Error formatting documents: {str(e)}")
            context = "No relevant docs were retrieved"
            context_metadata = "No data retrieved"
        
        return jsonify({
            "status": "success",
            "context": context,
            "context_metadata": context_metadata
        }), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error in retrieve_documents: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route('/delete_vectors', methods=['POST'])
def delete_vectors():
    file_deletion_status = []
    try:
        data = request.get_json()
        logger.info("===== Delete Vectors Request =====")
        logger.info(f"Request data: {data}")
        logger.info("----------------------------------------------------------------")
        
        user_id = data['user_id']
        file_related_data = data['file_related_data']
        embedding_model = data['embedding_model']

        # Normalize to list of document_id strings
        document_ids = []
        if isinstance(file_related_data, dict) and 'document_id' in file_related_data:
            document_ids = [str(file_related_data['document_id'])]
        elif isinstance(file_related_data, list):
            for item in file_related_data:
                if isinstance(item, dict) and 'document_id' in item:
                    document_ids.append(str(item['document_id']))
                else:
                    document_ids.append(str(item))
        else:
            document_ids = [str(file_related_data)]

        for doc_id in document_ids:
            message, ok = delete_document_from_qdrant(
                user_id=user_id,
                doc_id=doc_id,
                embedding_model=embedding_model
            )
            file_deletion_status.append({
                "user_id": user_id,
                "document_id": doc_id,
                "status": "success" if ok else "error",
                "message": message
            })
        
        logger.info(f"File deletion status: {file_deletion_status}")
        
        return jsonify({
            "status": "success",
            "message": "Vectors deleted successfully",
            "file_deletion_status": file_deletion_status
        }), 200
    except Exception as e:
        logger.error(f"Error in delete_vectors: {str(e)}")
        return jsonify({
            "status": "error",
            "file_deletion_status": file_deletion_status or [],
            "message": str(e)
        }), 400

@app.route('/remove_all_vectors', methods=['POST'])
def remove_all_vectors():
    try:
        logger.info("===== Remove All Vectors Request =====")
        data = request.get_json()
        user_id = data['user_id']
        embeddings_model = data['embeddings_model']
        logger.info(f"Request data: {data}")
        logger.info("----------------------------------------------------------------")

        message, ok = delete_collection_from_qdrant(user_id, embeddings_model)
        if ok:
            return jsonify({
                "status": "success",
                "message": message
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": message
            }), 500
    except Exception as e:
        logger.error(f"Error in remove_all_vectors: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route('/retrieve_documents_v2', methods=['POST'])
def retrieve_documents_v2():
    try:
        data = request.get_json()
        logger.info("===== Retrieve Documents Request =====")
        logger.info(f"Request data: {data}")
        logger.info("----------------------------------------------------------------")
        
        # Extract parameters from request
        query = data['query']
        user_id = data.get('user_id')
        document_id_list = data.get('document_id_list')
        score_threshold = data.get('score_threshold', 0.7)  # Default threshold
        k_value = data.get('k_value', 5)  # Default number of results
        llm_provider = data.get('llm_provider', 'openai')
        api_key = data['api_key']
        embedding_model = data['embedding_model']
        chat_model = data['chat_model']

        collection_name = build_collection_name(user_id, embedding_model)
        
        logger.info("Starting document retrieval...")
        # Call the document retriever
        results = doc_retriever(
            query=query,
            user_id=user_id,
            document_id_list=document_id_list,
            score_threshold=score_threshold,
            k_value=k_value,
            llm_provider=llm_provider,
            api_key=api_key,
            embedding_model=embedding_model,
            collection_name=collection_name,
            chat_model=chat_model
        )
        
        retrieved_docs = results['results']
        logger.info(f"Number of retrieved documents: {len(retrieved_docs) if retrieved_docs else 0}")
        
        # Convert Document objects to dictionaries with content and metadata
        if retrieved_docs and len(retrieved_docs) > 0:
            serialized_docs = []
            for doc in retrieved_docs:
                serialized_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            return jsonify({
                "status": "success",
                "documents": serialized_docs,
                "count": len(retrieved_docs)
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "No relevant docs were retrieved"
            }), 400
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error in retrieve_documents: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=2100, use_reloader=False)
