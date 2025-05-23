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
from batch_embed import EmbeddingQueue
from llm_provider import get_embeddings_model
from doc_retrieval import doc_retriever
from utils import format_docs, get_metadata_from_docs
from delete_vectors import delete_vectors_from_db, delete_all_vectors_from_db
from langchain_chroma import Chroma

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configure paths
BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_PATH = os.path.join(BASE_DIR, "VECTOR_DB")  # This will be /app/VECTOR_DB in container

# Ensure the vector store directory exists
os.makedirs(VECTORSTORE_PATH, exist_ok=True)

logger.info(f"Vector store path: {VECTORSTORE_PATH}")

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
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        split_docs = text_splitter.create_documents(normalized_doc)
        logger.info(f"Number of split documents: {len(split_docs)}")
        
        split_documents_with_metadata = [
            Document(page_content=document.page_content, metadata={"user_id": user_id, "doc_id": doc_id})
            for document in split_docs
        ]
        
        split_docs_length = len(split_documents_with_metadata)
        logger.info(f"Split documents with metadata length: {split_docs_length}")
        
        user_vector_store = os.path.join(VECTORSTORE_PATH, f"{user_id}",f"{embeddings_model}")
        # os.makedirs(user_vector_store, exist_ok=True)
        # os.chmod(user_vector_store, 0o777)  # Give full permissions to ensure write access
        
        # Create OpenAIEmbeddings instance
        embedding_function = get_embeddings_model(llm_provider, embeddings_model, api_key)
        logger.info(f"Embedding function created: {embedding_function}")

        # Calculate batch size based on token count
        if total_tokens_count > max_token_per_min:
            batch_size = split_docs_length // ((total_tokens_count // max_token_per_min) + 1)
        else:
            batch_size = split_docs_length
        logger.info(f"Calculated batch size: {batch_size}")
        
        # Initialize and start the queue
        embedding_queue = EmbeddingQueue(
            max_tokens_per_min=max_token_per_min,
            vectorstore_path=user_vector_store,
            embeddings=embedding_function,
            max_workers=3
        )
        logger.info("Embedding queue initialized")

        try:
            logger.info("Starting document processing...")
            processor_thread = embedding_queue.start_processing(split_documents_with_metadata, batch_size)
            processor_thread.join()  # Wait for all processing to complete
            
            if embedding_queue.processing_complete:
                logger.info("Processing completed successfully")
                return jsonify({
                    "status": "success",
                    "message": "All documents have been processed and stored in the vector database"
                }), 200
            logger.warning("Processing did not complete successfully")
            return jsonify({
                "status": "error",
                "message": "Processing was not completed successfully"
            }), 500
            
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
            embedding_queue.stop()
            processor_thread.join()
            return jsonify({
                "status": "error",
                "message": "Processing was interrupted"
            }), 500
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            embedding_queue.stop()
            processor_thread.join()
            return jsonify({
                "status": "error",
                "message": f"Error during processing: {str(e)}"
            }), 500
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
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

        user_vector_store = os.path.join(VECTORSTORE_PATH, f"{user_id}",f"{embedding_model}")
        
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
            vectorstore_path=user_vector_store,
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
    try:
        data = request.get_json()
        logger.info("===== Delete Vectors Request =====")
        logger.info(f"Request data: {data}")
        logger.info("----------------------------------------------------------------")
        
        user_id = data['user_id']
        file_related_data = data['file_related_data']
        llm_provider = data.get('llm_provider')
        api_key = data['api_key']
        embedding_model = data['embedding_model']
        
        file_deletion_status=[]
        user_vector_store = os.path.join(VECTORSTORE_PATH, f"{user_id}",f"{embedding_model}")
        for file_data in file_related_data:
            document_id = file_data["document_id"]
            logger.info(f"Processing document ID: {document_id}")

            if document_id is None:
                logger.error("Error: document_id is None")
                return jsonify({'error': 'the document_id cannot be None'}), 400
            
            document_id=str(document_id)
            if not document_id.isdigit():
                logger.error("Error: document_id is not an integer")
                return jsonify({'error': 'the document_id must be an integer'}), 400

            document_id_list = [document_id]
            logger.info(f"Deleting vectors for document ID: {document_id}")
            # Delete vectors from the vector database
            deletion_result = delete_vectors_from_db(user_id, document_id_list, user_vector_store, llm_provider, api_key, embedding_model)
            file_deletion_status.append(deletion_result)
            logger.info(f"Deletion result: {deletion_result}")
            
        logger.info("All vectors deleted successfully")
        return jsonify({
            "status": "success",
            "message": "Vectors deleted successfully",
            "file_deletion_status": file_deletion_status
        }), 200
    except Exception as e:
        logger.error(f"Error in delete_vectors: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


@app.route('/remove_all_vectors', methods=['POST'])
def remove_all_vectors():
    try:
        logger.info("===== Remove All Vectors Request =====")
        data = request.get_json()
        user_id = data['user_id']
        embeddings_model = data['embeddings_model']
        api_key = data['api_key']
        llm_provider = data['llm_provider']
        logger.info(f"Request data: {data}")
        logger.info("----------------------------------------------------------------")

        user_vector_store = os.path.join(VECTORSTORE_PATH, f"{user_id}",f"{embeddings_model}")
        try:
            # Check if directory exists before proceeding
            if os.path.exists(user_vector_store):
                delete_all_vectors_from_db(user_id, user_vector_store, llm_provider, api_key, embeddings_model)
                logger.info("Old vectors deleted successfully from vector store.")
                return jsonify({
                    "status": "success",
                    "message": "All old vectors of same model removed successfully"
                }), 200
            else:
                logger.info("No existing vector store found for this model.")
                return jsonify({
                    "status": "success",
                    "message": "No existing vectors found for this model"
                }), 200
                
        except Exception as e:
            logger.error(f"Error in remove_all_vectors: {str(e)}")
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 400
            
    except Exception as e:
        logger.error(f"Error in remove_all_vectors: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=2100, use_reloader=False)
