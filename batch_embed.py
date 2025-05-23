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
        try:
            return vector_store.add_documents(documents=documents)
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def process_batch(self, batch, batch_number):
        try:
            logger.info(f"Processing batch {batch_number} of {self.total_batches}")
            logger.info(f"Batch size: {len(batch)} documents")
            
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
            
            logger.info(f"Successfully processed batch {batch_number}")
            
            # Set processing_complete if this is the last batch
            if batch_number == self.total_batches:
                self.processing_complete = True
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_number}: {str(e)}")
            logger.info("Retrying in 15 seconds...")
            time.sleep(15)
            try:
                vector_store = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=self.vectorstore_path
                )
                self.add_documents_with_retry(vector_store, batch)
                logger.info(f"Successfully processed batch {batch_number} after retry")
                
                # Set processing_complete if this is the last batch
                if batch_number == self.total_batches:
                    self.processing_complete = True
                    
            except Exception as e:
                logger.error(f"Failed to process batch {batch_number} after retry: {str(e)}")
                raise
    
    def queue_processor(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                batch = self.queue.get(timeout=1)
                self.process_batch(batch, self.current_batch)
                self.current_batch += 1
                
                if not self.queue.empty():
                    logger.info(f"Waiting {45} seconds before processing next batch...")
                    logger.info(f"Remaining batches in queue: {self.queue.qsize()}")
                    time.sleep(45)
                
            except Empty:
                if self.current_batch > self.total_batches:
                    break  # Exit the loop if all batches are processed
                continue
            except Exception as e:
                logger.error(f"Unexpected error in queue processor: {str(e)}")
                self.stop_event.set()
                raise
    
    def start_processing(self, documents, batch_size):
        if not documents:
            raise ValueError("No documents provided for processing")
            
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
