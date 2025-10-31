from tenacity import retry, stop_after_attempt, wait_exponential
from qdrant_client.models import PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
import time
import hashlib

from vector_config.qdrant_client import get_collection_name, ensure_collection_exists   

class QdrantVectorStore:
    def __init__(self, client, embeddings, embedding_model):
        self.client = client
        self.embeddings = embeddings
        self.embedding_model = embedding_model

    @retry(
        stop=stop_after_attempt(3),  # Reduced attempts for faster failure detection
        wait=wait_exponential(multiplier=1, min=5, max=30)  # Faster retry for MistralAI
    )
    def add_documents_with_retry(self, documents, user_id):
        """Add documents to Qdrant with retry logic and validation"""
        try:
            # Validate documents before processing
            valid_documents = []
            for i, doc in enumerate(documents):
                if not doc.page_content or doc.page_content.strip() == "":
                    print(f"⚠️ Warning: Document {i} has empty content, skipping...")
                    continue
                
                # Check for very long content (Mistral has 8192 token limit)
                doc_tokens = self._calculate_document_tokens([doc])
                if doc_tokens > 8000:  # Leave some buffer
                    print(f"⚠️ Warning: Document {i} has {doc_tokens} tokens (too long), truncating...")
                    # Truncate the content to fit within limits
                    try:
                        # Try to get the model name from embeddings
                        if hasattr(self.embeddings, 'model_name'):
                            model_name = self.embeddings.model_name
                        elif hasattr(self.embeddings, 'model'):
                            model_name = self.embeddings.model
                        else:
                            model_name = "text-embedding-ada-002"
                        
                        import tiktoken
                        enc = tiktoken.encoding_for_model(model_name)
                    except Exception as e:
                        print(f"⚠️ Warning: Could not load tokenizer, using cl100k_base")
                        import tiktoken
                        enc = tiktoken.get_encoding("cl100k_base")
                    
                    tokens = enc.encode(doc.page_content)
                    if len(tokens) > 8000:
                        truncated_tokens = tokens[:8000]
                        doc.page_content = enc.decode(truncated_tokens)
                        print(f"✅ Truncated document {i} to {len(truncated_tokens)} tokens")
                
                valid_documents.append(doc)
            
            if not valid_documents:
                raise ValueError("No valid documents to process after validation")
            
            print(f"📝 Processing {len(valid_documents)} valid documents out of {len(documents)} total")
            
            # Add some debugging info
            for i, doc in enumerate(valid_documents[:3]):  # Show first 3 documents
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                print(f"Document {i} preview: {repr(preview)}")
            
            collection_name = get_collection_name(user_id, self.embedding_model)
            
            # Test embedding connection before processing
            self._test_embedding_connection()
            
            ensure_collection_exists(self.client, collection_name, self.embeddings)
            
            # Generate embeddings for documents with better error handling
            texts = [doc.page_content for doc in valid_documents]
            
            try:
                print(f"🔄 Generating embeddings for {len(texts)} documents...")
                print(f"   Using embedding model: {self.embedding_model}")
                print(f"   Provider: {type(self.embeddings).__name__}")
                
                # Add more detailed error handling for MistralAI
                if "MistralAI" in str(type(self.embeddings)):
                    print(f"   MistralAI configuration check:")
                    print(f"     - API Key present: {'Yes' if hasattr(self.embeddings, 'mistral_api_key') and self.embeddings.mistral_api_key else 'No'}")
                    print(f"     - Model: {getattr(self.embeddings, 'model', 'Unknown')}")
                
                # For MistralAI, try with smaller batches if the batch is too large
                if "MistralAI" in str(type(self.embeddings)) and len(texts) > 5:
                    print(f"   ⚠️  Large batch detected ({len(texts)} documents) for MistralAI")
                    print(f"   🔄 Attempting to process in smaller chunks...")
                    
                    embeddings_list = []
                    chunk_size = 3  # Very small chunks for MistralAI
                    
                    for i in range(0, len(texts), chunk_size):
                        chunk = texts[i:i + chunk_size]
                        print(f"   Processing chunk {i//chunk_size + 1}/{(len(texts) + chunk_size - 1)//chunk_size} ({len(chunk)} documents)")
                        
                        try:
                            chunk_embeddings = self.embeddings.embed_documents(chunk)
                            embeddings_list.extend(chunk_embeddings)
                            print(f"   ✅ Chunk {i//chunk_size + 1} successful")
                            
                            # Add small delay between chunks for MistralAI
                            if i + chunk_size < len(texts):
                                time.sleep(2)
                                
                        except Exception as chunk_error:
                            print(f"   ❌ Chunk {i//chunk_size + 1} failed: {str(chunk_error)}")
                            raise chunk_error
                    
                    print(f"✅ Successfully generated embeddings for {len(embeddings_list)} documents in chunks")
                else:
                    embeddings_list = self.embeddings.embed_documents(texts)
                    print(f"✅ Successfully generated embeddings for {len(embeddings_list)} documents")
                
            except Exception as embedding_error:
                print(f"❌ Error generating embeddings: {str(embedding_error)}")
                print(f"   Document count: {len(texts)}")
                print(f"   First document preview: {texts[0][:100] if texts else 'No text'}...")
                print(f"   Error type: {type(embedding_error).__name__}")
                
                # Enhanced error analysis
                error_str = str(embedding_error).lower()
                if "rate" in error_str or "429" in error_str:
                    print(f"⚠️  Rate limit detected. Consider reducing batch size or waiting longer.")
                elif "401" in error_str or "unauthorized" in error_str:
                    print(f"⚠️  Authentication error. Check your API key.")
                elif "403" in error_str or "forbidden" in error_str:
                    print(f"⚠️  Permission error. Check your API key permissions.")
                elif "timeout" in error_str:
                    print(f"⚠️  Timeout error. Consider reducing batch size.")
                elif "connection" in error_str:
                    print(f"⚠️  Connection error. Check your internet connection.")
                elif "mistral" in error_str:
                    print(f"⚠️  MistralAI specific error. Check API key and model configuration.")
                
                # Try to get more details about the error
                if hasattr(embedding_error, 'response'):
                    print(f"   HTTP Status: {getattr(embedding_error.response, 'status_code', 'Unknown')}")
                    print(f"   Response: {getattr(embedding_error.response, 'text', 'No response text')}")
                
                raise embedding_error
            
            # Prepare points for Qdrant
            points = []
            for i, (doc, embedding) in enumerate(zip(valid_documents, embeddings_list)):
                # Generate a unique integer ID for Qdrant
                unique_string = f"{user_id}_{doc.metadata['doc_id']}_{i}_{time.time()}"
                point_id = int(hashlib.md5(unique_string.encode()).hexdigest()[:16], 16)
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": doc.page_content,
                        "user_id": str(user_id),
                        "doc_id": doc.metadata["doc_id"],
                        "embedding_model": self.embedding_model,
                        "metadata": doc.metadata
                    }
                )
                points.append(point)
            
            # Upsert points to Qdrant
            print(f"📤 Upserting {len(points)} points to collection: {collection_name}")
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            print(f"✅ Successfully added {len(points)} documents to Qdrant")
            return True
            
        except UnexpectedResponse as e:
            print(f"❌ Qdrant server error: {str(e)}")
            print(f"Collection: {collection_name}")
            print(f"Points count: {len(points) if 'points' in locals() else 'N/A'}")
            raise
        except Exception as e:
            print(f"❌ General error adding documents to Qdrant: {str(e)}")
            print(f"Collection: {collection_name}")
            print(f"Points count: {len(points) if 'points' in locals() else 'N/A'}")
            raise
    
    def _test_embedding_connection(self):
        """Test the embedding connection with a simple query"""
        try:
            print(f"🔍 Testing embedding connection...")
            test_text = "test"
            test_embedding = None
            
            # Try different embedding methods based on the provider
            try:
                # Try embed_query first (most common)
                test_embedding = self.embeddings.embed_query(test_text)
                print(f"✅ Embedding connection test successful using embed_query")
                return True
            except Exception as e1:
                print(f"   ⚠️ embed_query failed: {str(e1)}")
                
                try:
                    # Try embed_documents for batch embedding
                    test_embedding = self.embeddings.embed_documents([test_text])
                    if test_embedding and len(test_embedding) > 0:
                        print(f"✅ Embedding connection test successful using embed_documents")
                        return True
                    else:
                        print(f"❌ embed_documents returned empty result")
                        raise Exception("Empty embedding result")
                except Exception as e2:
                    print(f"   ⚠️ embed_documents failed: {str(e2)}")
                    
                    try:
                        # Try aeval for some providers
                        test_embedding = self.embeddings.aeval(test_text)
                        print(f"✅ Embedding connection test successful using aeval")
                        return True
                    except Exception as e3:
                        print(f"   ⚠️ aeval failed: {str(e3)}")
                        
                        # If all methods fail, check if the embedding object has any embedding-related attributes
                        embedding_methods = [method for method in dir(self.embeddings) if 'embed' in method.lower()]
                        if embedding_methods:
                            print(f"   📋 Available embedding methods: {embedding_methods}")
                            print(f"   ❌ None of the standard embedding methods worked")
                            raise Exception(f"Embedding test failed. Available methods: {embedding_methods}")
                        else:
                            print(f"   ❌ No embedding methods found on object")
                            raise Exception("No embedding methods found on embedding object")
            
        except Exception as e:
            print(f"❌ Embedding connection test failed: {str(e)}")
            print(f"   This suggests an issue with your API key or model configuration")
            print(f"   Error type: {type(e).__name__}")
            raise e
    
    def _calculate_document_tokens(self, documents):
        """Calculate embedding tokens for different embedding models."""
        try:
            # Handle different embedding models
            if self.embedding_model in ['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large']:
                # OpenAI models - use tiktoken
                try:
                    import tiktoken
                    enc = tiktoken.encoding_for_model(self.embedding_model)
                    total_tokens = sum(len(enc.encode(text.page_content)) for text in documents)
                except Exception as tiktoken_error:
                    print(f"⚠️ Tiktoken error for {self.embedding_model}: {str(tiktoken_error)}")
                    print("   Falling back to character-based estimation...")
                    total_tokens = sum(len(text.page_content) // 4 for text in documents)
                
            elif self.embedding_model in ['mistral-embed', 'mistral-embed-v2']:
                # Mistral models - use approximate token counting
                total_tokens = sum(len(text.page_content) // 4 for text in documents)
                
            else:
                # For unknown models, use a conservative approximation
                print(f"⚠️ Unknown embedding model '{self.embedding_model}', using conservative token estimation")
                total_tokens = sum(len(text.page_content) // 4 for text in documents)
            
            return total_tokens
            
        except Exception as e:
            print(f"⚠️ Error calculating tokens for {self.embedding_model}: {str(e)}")
            print("   Using fallback token estimation...")
            
            # Fallback: use character-based estimation
            total_tokens = sum(len(text.page_content) // 4 for text in documents)
            return total_tokens
