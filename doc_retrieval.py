from typing import Any, Dict, List

from llm_provider import get_embeddings_model, get_chat_model
from doc_retrieve import get_vector_store, doc_retrieval


def doc_retriever(
    *,
    query: str,
    user_id: str,
    document_id_list: List[str],
    score_threshold: float,
    k_value: int,
    llm_provider: str,
    api_key: str,
    embedding_model: str,
    collection_name: str,
    chat_model: str,
) -> Dict[str, Any]:
    """Compatibility wrapper for the app to use the updated retrieval flow.

    Builds embeddings and chat model, creates the vector store, invokes the
    updated retrieval in `doc_retrieve.doc_retrieval`, and returns a dict with
    key `results` as expected by the Flask route.
    """

    # Build embeddings and chat model
    embeddings = get_embeddings_model(llm_provider, embedding_model, api_key)
    llm = get_chat_model(llm_provider, chat_model, api_key)

    # Build vector store from Qdrant (via updated helper)
    vectorstore, _ = get_vector_store(embeddings, user_id, embedding_model)

    # Run retrieval using the new implementation
    documents = doc_retrieval(
        vectorstore=vectorstore,
        user_id=user_id,
        document_id_list=document_id_list,
        current_threshold=score_threshold,
        k_value=k_value,
        question=query,
        llm=llm,
        embedding_model=embedding_model,
    )

    return {"results": documents}


