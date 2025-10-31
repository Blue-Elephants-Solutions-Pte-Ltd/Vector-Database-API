from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from langchain_qdrant import QdrantVectorStore
from vector_config.qdrant_client import get_qdrant_client, get_collection_name
from langchain_core.documents import Document

# Get embeddings from vector store
def get_vector_store(embeddings, user_id, embedding_model):
    try:
        client = get_qdrant_client()
        collection_name = get_collection_name(user_id, embedding_model)

        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )

        # Get collection info to count documents
        collection_info = client.get_collection(collection_name)
        chunk_count = collection_info.points_count
        print(f"Number of chunks in Qdrant collection: {chunk_count}")

        return vectorstore, chunk_count

    except Exception as e:
        print(f"❌ Error getting vector store: {str(e)}")
        # Return empty vector store if collection doesn't exist
        client = get_qdrant_client()
        collection_name = get_collection_name(user_id, embedding_model)
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )
        return vectorstore, 0

def standard_retriever(base_retriever, user_id, document_id_list, current_threshold, k_value, question, embedding_model):
    start = time.time()
    # Get the Qdrant client and collection name
    client = get_qdrant_client()
    collection_name = get_collection_name(user_id, embedding_model)
    
    # Get embeddings from the base retriever's vector store
    embeddings = base_retriever.vectorstore.embeddings
    
    # Get question embedding
    question_embedding = embeddings.embed_query(question)
    
    # Search directly in Qdrant for better control
    search_results = client.search(
        collection_name=collection_name,
        query_vector=question_embedding,
        query_filter={
            "must": [
                {"key": "user_id", "match": {"value": str(user_id)}},
                {"key": "doc_id", "match": {"any": document_id_list}}
            ]
        },
        limit=int(5),
        with_payload=True,
        score_threshold=float(current_threshold)
    )
    
    # Convert Qdrant results to LangChain documents
    standard_retriever_docs = []
    for result in search_results:
        # Create document with full content
        doc = Document(
            page_content=result.payload.get('text', ''),
            metadata={
                'user_id': result.payload.get('user_id', user_id),
                'doc_id': result.payload.get('doc_id', ''),
                'score': result.score
            }
        )
        standard_retriever_docs.append(doc)
    
    # If no results with threshold, try without threshold
    if not standard_retriever_docs:
        search_results = client.search(
            collection_name=collection_name,
            query_vector=question_embedding,
            query_filter={
                "must": [
                    {"key": "user_id", "match": {"value": str(user_id)}},
                    {"key": "doc_id", "match": {"any": document_id_list}}
                ]
            },
            limit=int(k_value),
            with_payload=True
        )
        
        for result in search_results:
            doc = Document(
                page_content=result.payload.get('text', ''),
                metadata={
                    'user_id': result.payload.get('user_id', user_id),
                    'doc_id': result.payload.get('doc_id', ''),
                    'score': result.score
                }
            )   
            standard_retriever_docs.append(doc)
    print(f"total time taken for standard_retriever: {time.time() - start:.4f} seconds")
    return standard_retriever_docs

def contextual_retriever(Contextual_base_retriever, user_id, document_id_list, question, llm, embedding_model):
    start = time.time()
    document_id_list = [str(d) for d in document_id_list]
    
    # Get Qdrant client and collection name
    client = get_qdrant_client()
    collection_name = get_collection_name(user_id, embedding_model)
    
    # Get embeddings for dense vector search
    embeddings = Contextual_base_retriever.vectorstore.embeddings
    question_embedding = embeddings.embed_query(question)
    
    # Common filter conditions
    filter_conditions = {
        "must": [
            {"key": "user_id", "match": {"value": str(user_id)}},
            {"key": "doc_id", "match": {"any": document_id_list}}
        ]
    }
    
    # Step 1: Dense Vector Search (Semantic)
    try:
        dense_results = client.search(
            collection_name=collection_name,
            query_vector=question_embedding,
            query_filter=filter_conditions,
            limit=8,  # Get more candidates for hybrid
            with_payload=True,
            score_threshold=0.1
        )
    except Exception as e:
        dense_results = []
    
    # Step 2: Keyword-based Search (using text matching)
    try:
        # Use text matching for keyword search instead of sparse vectors
        keyword_results = client.search(
            collection_name=collection_name,
            query_vector=question_embedding,  # Still use embeddings for consistency
            query_filter={
                "must": [
                    {"key": "user_id", "match": {"value": str(user_id)}},
                    {"key": "doc_id", "match": {"any": document_id_list}},
                    {"key": "text", "match": {"text": question}}  # Text matching for keywords
                ]
            },
            limit=8,  # Get more candidates for hybrid
            with_payload=True,
            score_threshold=0.05  # Lower threshold for keyword matching
        )
    except Exception as e:
        try:
            # Fallback: use broader search without text matching
            keyword_results = client.search(
                collection_name=collection_name,
                query_vector=question_embedding,
                query_filter=filter_conditions,
                limit=6,  # Fewer results for fallback
                with_payload=True,
                score_threshold=0.15  # Higher threshold for fallback
            )
        except Exception as fallback_error:
            keyword_results = []
    
    # Step 3: Combine and Deduplicate Results
    combined_results = []
    seen_ids = set()
    
    # Add dense results first (higher priority for semantic understanding)
    for result in dense_results:
        if result.id not in seen_ids:
            combined_results.append(result)
            seen_ids.add(result.id)
    
    # Add keyword results (for keyword coverage)
    for result in keyword_results:
        if result.id not in seen_ids:
            combined_results.append(result)
            seen_ids.add(result.id)
    
    # Step 4: Convert to Documents for LangChain
    documents = []
    for result in combined_results:
        doc = Document(
            page_content=result.payload.get('text', ''),
            metadata={
                'user_id': result.payload.get('user_id', user_id),
                'doc_id': result.payload.get('doc_id', ''),
                'score': result.score,
                'search_type': 'hybrid'
            }
        )
        documents.append(doc)
    
    # Step 5: Apply Contextual Compression
    try:
        # Apply compression directly to documents without custom retriever
        compressor = LLMChainExtractor.from_llm(llm)
        
        # Compress each document individually
        compressed_docs = []
        for doc in documents[:8]:  # Process top 8 documents
            try:
                # Create a simple prompt for compression
                compression_prompt = f"""Given the following question and document content, extract the most relevant information that answers the question.

                    Question: {question}

                    Document Content:
                    {doc.page_content}

                    Relevant Information:"""

                # Use LLM to compress the document
                response = llm.invoke(compression_prompt)
                compressed_content = response.content if hasattr(response, 'content') else str(response)
                
                # Create compressed document
                compressed_doc = Document(
                    page_content=compressed_content,
                    metadata={
                        **doc.metadata,
                        'original_content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                        'compression_type': 'llm_extraction'
                    }
                )
                compressed_docs.append(compressed_doc)
                
            except Exception as doc_error:
                # Keep original document if compression fails
                compressed_docs.append(doc)
        
        contextual_retriever_docs = compressed_docs[:5]  # Return top 5 compressed results
        
    except Exception as e:
        # Sort by score and return top results
        sorted_docs = sorted(documents, key=lambda x: x.metadata.get('score', 0), reverse=True)
        contextual_retriever_docs = sorted_docs[:5]  # Return top 5 results by score
    print(f"total time taken for contextual_retriever: {time.time() - start:.4f} seconds")
    return contextual_retriever_docs

def enhanced_retriever(base_retriever, user_id, document_id_list, current_threshold, k_value, question, embedding_model):
    start = time.time()
    client = get_qdrant_client()
    collection_name = get_collection_name(user_id, embedding_model)
    embeddings = base_retriever.vectorstore.embeddings
    question_embedding = embeddings.embed_query(question)
    
    # Strategy 1: Semantic Search (Vector Similarity)
    semantic_results = client.search(
        collection_name=collection_name,
        query_vector=question_embedding,
        query_filter={
            "must": [
                {"key": "user_id", "match": {"value": str(user_id)}},
                {"key": "doc_id", "match": {"any": document_id_list}}
            ]
        },
        limit=10,
        with_payload=True,
        score_threshold=0.1
    )
    
    # Strategy 2: Keyword Search (Text Matching)
    keyword_results = client.search(
        collection_name=collection_name,
        query_vector=question_embedding,
        query_filter={
            "must": [
                {"key": "user_id", "match": {"value": str(user_id)}},
                {"key": "doc_id", "match": {"any": document_id_list}},
                {"key": "text", "match": {"text": question}}
            ]
        },
        limit=8,
        with_payload=True,
        score_threshold=0.05
    )
    
    # Strategy 3: Broader Context Search (Lower threshold)
    context_results = client.search(
        collection_name=collection_name,
        query_vector=question_embedding,
        query_filter={
            "must": [
                {"key": "user_id", "match": {"value": str(user_id)}},
                {"key": "doc_id", "match": {"any": document_id_list}}
            ]
        },
        limit=15,
        with_payload=True,
        score_threshold=0.05  # Very low threshold for broader context
    )
    
    # Combine and deduplicate results
    all_results = []
    seen_ids = set()
    
    # Add semantic results first (highest relevance)
    for result in semantic_results:
        if result.id not in seen_ids:
            all_results.append(result)
            seen_ids.add(result.id)
    
    # Add keyword results
    for result in keyword_results:
        if result.id not in seen_ids:
            all_results.append(result)
            seen_ids.add(result.id)
    
    # Add context results
    for result in context_results:
        if result.id not in seen_ids:
            all_results.append(result)
            seen_ids.add(result.id)
    
    # Convert to documents
    documents = []
    for result in all_results:
        doc = Document(
            page_content=result.payload.get('text', ''),
            metadata={
                'user_id': result.payload.get('user_id', user_id),
                'doc_id': result.payload.get('doc_id', ''),
                'score': result.score,
                'search_type': 'enhanced_multi_strategy'
            }
        )
        documents.append(doc)
    
    print(f"total time taken for enhanced_retriever: {time.time() - start:.4f} seconds")
    return documents[:k_value]  # Return top k results

def hierarchical_retriever(base_retriever, user_id, document_id_list, current_threshold, k_value, question, embedding_model):
    start = time.time()
    client = get_qdrant_client()
    collection_name = get_collection_name(user_id, embedding_model)
    embeddings = base_retriever.vectorstore.embeddings
    question_embedding = embeddings.embed_query(question)
    
    all_documents = []
    seen_ids = set()
    
    # Level 1: High precision search (high threshold)
    high_precision = client.search(
        collection_name=collection_name,
        query_vector=question_embedding,
        query_filter={
            "must": [
                {"key": "user_id", "match": {"value": str(user_id)}},
                {"key": "doc_id", "match": {"any": document_id_list}}
            ]
        },
        limit=5,
        with_payload=True,
        score_threshold=0.3  # High threshold for precision
    )
    
    for result in high_precision:
        if result.id not in seen_ids:
            all_documents.append(Document(
                page_content=result.payload.get('text', ''),
                metadata={
                    'user_id': result.payload.get('user_id', user_id),
                    'doc_id': result.payload.get('doc_id', ''),
                    'score': result.score,
                    'search_level': 'high_precision'
                }
            ))
            seen_ids.add(result.id)
    
    # Level 2: Medium precision search
    if len(all_documents) < k_value:
        medium_precision = client.search(
            collection_name=collection_name,
            query_vector=question_embedding,
            query_filter={
                "must": [
                    {"key": "user_id", "match": {"value": str(user_id)}},
                    {"key": "doc_id", "match": {"any": document_id_list}}
                ]
            },
            limit=10,
            with_payload=True,
            score_threshold=0.15  # Medium threshold
        )
        
        for result in medium_precision:
            if result.id not in seen_ids and len(all_documents) < k_value:
                all_documents.append(Document(
                    page_content=result.payload.get('text', ''),
                    metadata={
                        'user_id': result.payload.get('user_id', user_id),
                        'doc_id': result.payload.get('doc_id', ''),
                        'score': result.score,
                        'search_level': 'medium_precision'
                    }
                ))
                seen_ids.add(result.id)
    
    # Level 3: Low precision search for broader context
    if len(all_documents) < k_value:
        low_precision = client.search(
            collection_name=collection_name,
            query_vector=question_embedding,
            query_filter={
                "must": [
                    {"key": "user_id", "match": {"value": str(user_id)}},
                    {"key": "doc_id", "match": {"any": document_id_list}}
                ]
            },
            limit=15,
            with_payload=True,
            score_threshold=0.05  # Low threshold for recall
        )
        
        for result in low_precision:
            if result.id not in seen_ids and len(all_documents) < k_value:
                all_documents.append(Document(
                    page_content=result.payload.get('text', ''),
                    metadata={
                        'user_id': result.payload.get('user_id', user_id),
                        'doc_id': result.payload.get('doc_id', ''),
                        'score': result.score,
                        'search_level': 'low_precision'
                    }
                ))
                seen_ids.add(result.id)
    
    print(f"total time taken for hierarchical_retriever: {time.time() - start:.4f} seconds")
    return all_documents

def hybrid_retriever(base_retriever, user_id, document_id_list, current_threshold, k_value, question, embedding_model):
    start = time.time()
    client = get_qdrant_client()
    collection_name = get_collection_name(user_id, embedding_model)
    embeddings = base_retriever.vectorstore.embeddings
    question_embedding = embeddings.embed_query(question)
    
    # Extract keywords from question for better matching
    keywords = extract_keywords(question)
    
    # Semantic search
    semantic_results = client.search(
        collection_name=collection_name,
        query_vector=question_embedding,
        query_filter={
            "must": [
                {"key": "user_id", "match": {"value": str(user_id)}},
                {"key": "doc_id", "match": {"any": document_id_list}}
            ]
        },
        limit=8,
        with_payload=True,
        score_threshold=0.1
    )
    
    # Keyword search with multiple keywords
    keyword_results = []
    for keyword in keywords[:3]:  # Use top 3 keywords
        try:
            results = client.search(
                collection_name=collection_name,
                query_vector=question_embedding,
                query_filter={
                    "must": [
                        {"key": "user_id", "match": {"value": str(user_id)}},
                        {"key": "doc_id", "match": {"any": document_id_list}},
                        {"key": "text", "match": {"text": keyword}}
                    ]
                },
                limit=5,
                with_payload=True,
                score_threshold=0.05
            )
            keyword_results.extend(results)
        except:
            continue
    
    # Combine and rank results
    all_results = semantic_results + keyword_results
    seen_ids = set()
    unique_results = []
    
    for result in all_results:
        if result.id not in seen_ids:
            unique_results.append(result)
            seen_ids.add(result.id)
    
    # Sort by score and return top k
    unique_results.sort(key=lambda x: x.score, reverse=True)
    
    documents = []
    for result in unique_results[:k_value]:
        documents.append(Document(
            page_content=result.payload.get('text', ''),
            metadata={
                'user_id': result.payload.get('user_id', user_id),
                'doc_id': result.payload.get('doc_id', ''),
                'score': result.score,
                'search_type': 'hybrid'
            }
        ))
    
    print(f"total time taken for hybrid_retriever: {time.time() - start:.4f} seconds")
    return documents

def extract_keywords(question):
    """Extract important keywords from the question"""
    # Simple keyword extraction - you can enhance this
    stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    words = question.lower().split()
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    return keywords[:5]  # Return top 5 keywords

def context_aware_retriever(base_retriever, user_id, document_id_list, current_threshold, k_value, question, embedding_model):
    start = time.time()
    client = get_qdrant_client()
    collection_name = get_collection_name(user_id, embedding_model)
    embeddings = base_retriever.vectorstore.embeddings
    question_embedding = embeddings.embed_query(question)
    
    # Get all documents for the specified document IDs first
    all_docs = client.scroll(
        collection_name=collection_name,
        scroll_filter={
            "must": [
                {"key": "user_id", "match": {"value": str(user_id)}},
                {"key": "doc_id", "match": {"any": document_id_list}}
            ]
        },
        limit=1000,  # Get all documents
        with_payload=True
    )
    
    # Perform semantic search on the retrieved documents
    search_results = client.search(
        collection_name=collection_name,
        query_vector=question_embedding,
        query_filter={
            "must": [
                {"key": "user_id", "match": {"value": str(user_id)}},
                {"key": "doc_id", "match": {"any": document_id_list}}
            ]
        },
        limit=k_value * 2,  # Get more candidates
        with_payload=True,
        score_threshold=0.05
    )
    
    # Group by document ID to get context
    doc_groups = {}
    for result in search_results:
        doc_id = result.payload.get('doc_id', '')
        if doc_id not in doc_groups:
            doc_groups[doc_id] = []
        doc_groups[doc_id].append(result)
    
    # Select best chunks from each document
    final_documents = []
    for doc_id, chunks in doc_groups.items():
        # Sort chunks by score and take the best ones
        chunks.sort(key=lambda x: x.score, reverse=True)
        best_chunks = chunks[:2]  # Take top 2 chunks from each document
        
        for chunk in best_chunks:
            final_documents.append(Document(
                page_content=chunk.payload.get('text', ''),
                metadata={
                    'user_id': chunk.payload.get('user_id', user_id),
                    'doc_id': chunk.payload.get('doc_id', ''),
                    'score': chunk.score,
                    'search_type': 'context_aware',
                    'document_context': f"From document {doc_id}"
                }
            ))
    
    # Sort by score and return top k
    final_documents.sort(key=lambda x: x.metadata['score'], reverse=True)
    
    print(f"total time taken for context_aware_retriever: {time.time() - start:.4f} seconds")
    return final_documents[:k_value]

def doc_retrieval(vectorstore, user_id, document_id_list, current_threshold, k_value, question, llm, embedding_model):
    """
    Retrieve documents using both standard and contextual retrievers in parallel.
    """
    base_retriever=vectorstore.as_retriever()

    # Use ThreadPoolExecutor to run both functions concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        contextual_future = executor.submit(
            enhanced_retriever,
            base_retriever, 
            user_id, 
            document_id_list, 
            current_threshold,
            k_value,
            question, 
            embedding_model
        )
        
        standard_future = executor.submit(
            standard_retriever,
            base_retriever, 
            user_id, 
            document_id_list, 
            current_threshold, 
            k_value, 
            question,
            embedding_model
        )
        
        # Wait for both tasks to complete and get results
        try:
            contextual_retriever_docs = contextual_future.result()
            standard_retriever_docs = standard_future.result()
        except Exception as e:
            print(f"Error during parallel retrieval: {e}")
            # Fallback to sequential execution if parallel fails
            contextual_retriever_docs = enhanced_retriever(
                base_retriever, user_id, document_id_list, current_threshold, k_value, question, embedding_model
            )
            standard_retriever_docs = standard_retriever(
                base_retriever, user_id, document_id_list, current_threshold, k_value, question, embedding_model
            )
    
    
    print(f"number of chunks from enhanced_retriever: {len(contextual_retriever_docs)}")
    print(f"number of chunks from standard_retriever: {len(standard_retriever_docs)}")
    
    # Combine results with deduplication
    combined_docs = []
    seen_content = set()
    
    # Add enhanced results first (higher quality due to multi-strategy)
    for doc in contextual_retriever_docs:
        content_hash = hash(doc.page_content[:100])  # Hash first 100 chars for deduplication
        if content_hash not in seen_content:
            combined_docs.append(doc)
            seen_content.add(content_hash)
    
    # Add standard results (for additional coverage)
    for doc in standard_retriever_docs:
        content_hash = hash(doc.page_content[:100])
        if content_hash not in seen_content:
            combined_docs.append(doc)
            seen_content.add(content_hash)
            
    print(f"number of chunks from combined_docs: {len(combined_docs)}")

    return combined_docs