def format_docs(docs):
    
    final = []
    for doc in docs:
        final.append(doc.page_content)
    
    return final

def get_metadata_from_docs(docs):
    metadata=[]
    for doc in docs:
        metadata.append(doc.metadata)

    # Extract unique doc_ids and user_id
    unique_doc_ids = list({entry["doc_id"] for entry in metadata})
    unique_user_ids = list({entry["user_id"] for entry in metadata})

    # Create the desired output format
    metadata = {"user_id": unique_user_ids[0], "doc_ids": unique_doc_ids}
    return metadata
