from langchain_chroma import Chroma
import time
from queue import Queue, Empty
from threading import Thread, Event
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)

############################################### Function to store vector ##############################################
class EmbeddingQueue:
    def __init__(self, max_tokens_per_min=2000000, vectorstore_path=None, embeddings=None, max_workers=3):
        self.queue = Queue()
        self.max_tokens_per_min = max_tokens_per_min
        self.vectorstore_path = vectorstore_path
        self.embeddings = embeddings
        self.stop_event = Event()
        self.current_batch = 1
        self.total_batches = 0
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_complete = False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def add_documents_with_retry(self, vector_store, documents):
        return vector_store.add_documents(documents=documents)
    
    def process_batch(self, batch, batch_number):
        try:
            print(f"\nProcessing batch {batch_number} of {self.total_batches}")
            print(f"Batch size: {len(batch)} documents")
            
            vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.vectorstore_path
            )
            
            # Split batch into smaller sub-batches for parallel processing
            sub_batch_size = max(1, len(batch) // self.max_workers)
            sub_batches = [batch[i:i + sub_batch_size] for i in range(0, len(batch), sub_batch_size)]
            
            # Process sub-batches in parallel
            futures = []
            for sub_batch in sub_batches:
                future = self.executor.submit(self.add_documents_with_retry, vector_store, sub_batch)
                futures.append(future)
            
            # Wait for all sub-batches to complete
            for future in futures:
                future.result()
            
            print(f"✅ Successfully processed batch {batch_number}")
            
            # Set processing_complete if this is the last batch
            if batch_number == self.total_batches:
                self.processing_complete = True
                
        except Exception as e:
            print(f"❌ Error processing batch {batch_number}: {str(e)}")
            print("Retrying in 15 seconds...")
            time.sleep(15)
            try:
                vector_store = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=self.vectorstore_path
                )
                self.add_documents_with_retry(vector_store, batch)
                print(f"✅ Successfully processed batch {batch_number} after retry")
                
                # Set processing_complete if this is the last batch
                if batch_number == self.total_batches:
                    self.processing_complete = True
                    
            except Exception as e:
                print(f"❌ Failed to process batch {batch_number} after retry: {str(e)}")
                raise
    
    def queue_processor(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                batch = self.queue.get(timeout=1)
                self.process_batch(batch, self.current_batch)
                self.current_batch += 1
                
                if not self.queue.empty():
                    print("\nWaiting 45 seconds before processing next batch...")
                    print(f"Remaining batches in queue: {self.queue.qsize()}")
                    time.sleep(45)
                
            except Empty:
                if self.current_batch > self.total_batches:
                    break  # Exit the loop if all batches are processed
                continue
            except Exception as e:
                print(f"Unexpected error in queue processor: {str(e)}")
                self.stop_event.set()
                raise
    
    def start_processing(self, documents, batch_size):
        self.total_batches = (len(documents) + batch_size - 1) // batch_size
        
        # Split documents into batches and add to queue
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.queue.put(batch)
        
        # Start processor thread
        processor_thread = Thread(target=self.queue_processor)
        processor_thread.start()
        
        return processor_thread
    
    def stop(self):
        self.stop_event.set()
        self.executor.shutdown(wait=True)


def vectorestore_function(split_documents_with_metadata, user_vector_store_path, embeddings, max_token_per_min, total_tokens_count):
    processor_thread = None  # Initialize processor_thread as None
    embedding_queue = None   # Initialize embedding_queue as None
    try:
        split_docs_length = len(split_documents_with_metadata)
        
        if total_tokens_count > max_token_per_min:
            batch_size = split_docs_length // ((total_tokens_count // max_token_per_min) + 1)
        else:
            batch_size = split_docs_length
        
        logger.info(f"Calculated batch size: {batch_size}")
        
        # Initialize and start the queue
        embedding_queue = EmbeddingQueue(
            max_tokens_per_min=max_token_per_min,
            vectorstore_path=user_vector_store_path,
            embeddings=embeddings,
            max_workers=3
        )

        try:
            logger.info("Starting document processing...")
            processor_thread = embedding_queue.start_processing(split_documents_with_metadata, batch_size)
            processor_thread.join()  # Wait for all processing to complete
        
            if embedding_queue.processing_complete:
                logger.info("Processing completed successfully")
                return "success", "All documents have been processed and stored in the vector database"
            logger.warning("Processing did not complete successfully")
            return "error", "Processing was not completed successfully"
            
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
            embedding_queue.stop()
            processor_thread.join()
            return "error", "Processing was interrupted"
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            embedding_queue.stop()
            processor_thread.join()
            return "error", f"Error during processing: {str(e)}"
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return "error", str(e)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return "error", str(e)