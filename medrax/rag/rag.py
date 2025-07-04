import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import Field
import gc
import sys
import signal
from contextlib import contextmanager

from pydantic import BaseModel, Field
from langchain_cohere import ChatCohere, CohereEmbeddings, CohereRerank
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseRetriever
from langchain.docstore.document import Document
from typing import Callable
from datasets import load_dataset
from tqdm import tqdm


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


@contextmanager
def timeout(seconds):
    """Context manager for timing out operations"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


def safe_extract_pdf_text(pdf_data, timeout_seconds=180):
    """Safely extract text from PDF with timeout and fallback handling"""
    try:
        with timeout(timeout_seconds):
            if isinstance(pdf_data, str):
                return pdf_data, {"file_type": "pdf"}
            elif hasattr(pdf_data, 'pages'):
                # pdfplumber PDF object
                pdf_text = ""
                num_pages = len(pdf_data.pages)
                
                for page in pdf_data.pages:
                    text = page.extract_text()
                    if text:
                        pdf_text += text + "\n"
                
                # Close the PDF to free resources
                pdf_data.close()
                
                return pdf_text.strip(), {"file_type": "pdf", "num_pages": num_pages}
            else:
                return None, None
                
    except TimeoutError:
        print(f"PDF processing timed out after {timeout_seconds} seconds")
        try:
            if hasattr(pdf_data, 'close'):
                pdf_data.close()
        except:
            pass
        return None, None
    except Exception as e:
        print(f"Error in PDF processing: {str(e)}")
        try:
            if hasattr(pdf_data, 'close'):
                pdf_data.close()
        except:
            pass
        return None, None


class RAGConfig(BaseModel):
    """Configuration class for RAG (Retrieval Augmented Generation) system.

    Attributes:
        model (str): Cohere model name for chat completion
        temperature (float): Sampling temperature between 0 and 1
        persist_dir (str): Directory to persist vector database
        embedding_model (str): Cohere model name for embeddings
        rerank_model (str): Cohere model name for reranking
        retriever_k (int): Number of documents to retrieve
        chunk_size (int): Size of text chunks for splitting
        chunk_overlap (int): Overlap between text chunks
        local_docs_dir (str): Directory for text files
        huggingface_datasets (List[str]): List of HuggingFace dataset names to load (e.g., ["MedRAG/textbooks", "VictorLJZ/medrax"])
        dataset_split (str): Split to use for HuggingFace datasets (default: "train")
    """

    model: str = Field(default="command-a-03-2025")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    pinecone_index_name: str = Field(default="medrax")
    embedding_model: str = Field(default="embed-v4.0")
    rerank_model: str = Field(default="rerank-v3.5")
    retriever_k: int = Field(default=5)
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    local_docs_dir: str = Field(default="medrax/rag/docs")
    huggingface_datasets: List[str] = Field(default_factory=lambda: ["MedRAG/textbooks"])
    dataset_split: str = Field(default="train")


class RerankingRetriever(BaseRetriever):
    """Custom retriever that wraps a document retrieval function with reranking.

    Attributes:
        get_relevant_docs_func (Callable): Function that retrieves relevant documents
    """

    get_relevant_docs_func: Callable[[str], List[Document]] = Field(...)

    def __init__(self, get_relevant_docs_func: Callable[[str], List[Document]]):
        """Initialize retriever with document retrieval function."""
        super().__init__(get_relevant_docs_func=get_relevant_docs_func)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query.

        Args:
            query (str): Search query

        Returns:
            List[Document]: Retrieved documents
        """
        return self.get_relevant_docs_func(query)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async retrieval not implemented."""
        raise NotImplementedError("Async retrieval not implemented")


class CohereRAG:
    """RAG system implementation using Cohere's models for embedding, reranking and chat.

    Attributes:
        config (RAGConfig): Configuration for the RAG system
        chat_model (ChatCohere): Cohere chat model
        embeddings (CohereEmbeddings): Cohere embeddings model
        reranker (CohereRerank): Cohere reranking model
        persist_dir (str): Directory for vector database
        memory (ConversationBufferMemory): Conversation memory
        vectorstore (Optional[Chroma]): Vector database for document storage
        local_docs_dir (str): Directory for text files
    """

    def __init__(self, config: RAGConfig = RAGConfig()):
        """Initialize RAG system with given configuration."""
        self.config = config
        self.chat_model = ChatCohere(model=config.model, temperature=config.temperature)
        self.embeddings = CohereEmbeddings(model=config.embedding_model)
        self.reranker = CohereRerank(model=config.rerank_model)
        self.local_docs_dir = config.local_docs_dir
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="result",
            input_key="query",
        )

        # Initialize Pinecone
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set. Please get a key from app.pinecone.io")
        self.pinecone = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = self.config.pinecone_index_name

        # Create or update Pinecone index and get the vectorstore
        self.vectorstore = self.get_or_create_vectorstore()

    def load_directory(self, directory_path: str) -> List[Document]:
        """Load and split all .txt files from a directory into documents.

        Args:
            directory_path (str): Path to directory containing text files

        Returns:
            List[Document]: List of processed documents

        Raises:
            ValueError: If directory does not exist
        """
        documents = []
        directory = Path(directory_path)

        if not directory.exists():
            raise ValueError(f"Directory {directory_path} does not exist")

        # Configure text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
        )

        # Process each document file (txt, pdf, docx)
        for file_pattern in ["**/*.txt", "**/*.pdf", "**/*.docx"]:
            for file_path in directory.glob(file_pattern):
                try:
                    # Select appropriate loader based on file extension
                    if file_path.suffix.lower() == ".txt":
                        loader = TextLoader(str(file_path))
                    elif file_path.suffix.lower() == ".pdf":
                        loader = PyPDFLoader(str(file_path))
                    elif file_path.suffix.lower() == ".docx":
                        loader = Docx2txtLoader(str(file_path))

                    docs = loader.load()

                    # Split documents first
                    split_docs = text_splitter.split_documents(docs)

                    # Add metadata to each document
                    metadata = {
                        "source": str(file_path),
                        "created_at": os.path.getctime(file_path),
                        "file_type": file_path.suffix.lower()[1:],
                    }

                    for doc in split_docs:
                        doc.metadata.update(metadata)

                    documents.extend(split_docs)
                    print(f"Loaded and split: {file_path}")

                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")

        return documents

    def get_or_create_vectorstore(self) -> PineconeVectorStore:
        """
        Connects to an existing Pinecone index. If the index is empty, it populates it with documents.
        The index must be created manually in the Pinecone console beforehand.
        """
        if self.index_name not in self.pinecone.list_indexes().names():
            raise ValueError(
                f"Index '{self.index_name}' not found in your Pinecone project. "
                f"Please create it manually in the Pinecone console. "
                f"For the free tier, use the 'aws-us-east-1' environment."
            )

        print(f"Connecting to existing Pinecone index: {self.index_name}")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=self.index_name, embedding=self.embeddings
        )

        # Check if the index is empty and needs to be populated
        try:
            # Get the index object directly
            index = self.pinecone.Index(self.index_name)
            
            # Get index stats
            stats = index.describe_index_stats()
            print(f"Index stats: {stats}")
            
            total_vectors = stats.get('total_vector_count', 0)
            print(f"Total vectors in index: {total_vectors}")
            
            if total_vectors == 0:
                print("Index is empty. Populating with documents...")
                documents = self._load_all_documents()
                if documents:
                    total_docs = len(documents)
                    print(
                        f"Adding {total_docs} documents to the index. This may take a while..."
                    )

                    # Batching mechanism to handle rate limits
                    batch_size = 50  # Process 50 documents per batch
                    for i in tqdm(
                        range(0, total_docs, batch_size),
                        desc="Adding documents to Pinecone",
                    ):
                        batch = documents[i : i + batch_size]
                        vectorstore.add_documents(batch)
                        # Removed rate limiting - process as fast as possible

                    print("Documents added successfully.")
                    
                    # Verify documents were added
                    final_stats = index.describe_index_stats()
                    final_count = final_stats.get('total_vector_count', 0)
                    print(f"Final vector count after adding documents: {final_count}")
                else:
                    print("Warning: No documents found to add to the new index.")
            else:
                print(f"Index already populated with {total_vectors} vectors.")
                
        except Exception as e:
            print(f"Error checking index stats: {e}")
            print("Proceeding without stats check...")

        return vectorstore

    def _load_all_documents(self) -> List[Document]:
        """Collect documents from all enabled sources."""
        all_documents = []
        
        # Load HuggingFace datasets
        for dataset_name in self.config.huggingface_datasets:
            print(f"Loading documents from HuggingFace dataset: {dataset_name}...")
            try:
                hf_docs = self.load_huggingface_dataset(dataset_name, self.config.dataset_split)
                all_documents.extend(hf_docs)
                print(f"Loaded {len(hf_docs)} documents from {dataset_name}")
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {str(e)}")
                continue

        # Load local documents
        if os.path.exists(self.local_docs_dir):
            print(f"Loading documents from local directory: {self.local_docs_dir}")
            local_docs = self.load_directory(self.local_docs_dir)
            all_documents.extend(local_docs)
            print(f"Loaded {len(local_docs)} documents from local directory")
        
        if not all_documents:
            print("Warning: No documents loaded. Please check your configuration.")
            
        return all_documents

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using vector similarity and reranking.

        Args:
            query (str): Search query

        Returns:
            List[Document]: Reranked relevant documents
        """
        # Get initial candidates using vector similarity
        docs = self.vectorstore.similarity_search(query, k=self.config.retriever_k * 2)

        # Rerank documents
        reranked = self.reranker.rerank(query=query, documents=[doc.page_content for doc in docs])

        # Return top k documents after reranking
        return [docs[result["index"]] for result in reranked[: self.config.retriever_k]]

    def initialize_rag(self, with_memory: bool = False) -> RetrievalQA:
        """Initialize RAG chain with optional conversation memory.

        Args:
            with_memory (bool): Whether to include conversation memory

        Returns:
            RetrievalQA: Configured RAG chain

        Raises:
            ValueError: If vectorstore not initialized
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Please add documents first.")

        # Create custom retriever
        retriever = RerankingRetriever(self.get_relevant_documents)

        # Configure chain parameters
        chain_kwargs = {
            "llm": self.chat_model,
            "chain_type": "stuff",
            "retriever": retriever,
            "return_source_documents": True,
            "verbose": True,
        }

        if with_memory:
            chain_kwargs["memory"] = self.memory

        return RetrievalQA.from_chain_type(**chain_kwargs)

    def load_huggingface_dataset(self, dataset_name: str, split: str = "train") -> List[Document]:
        """Load dataset from Hugging Face with batch processing for memory efficiency.

        Args:
            dataset_name (str): Name of the dataset on Hugging Face (e.g., "MedRAG/textbooks", "VictorLJZ/medrax")
            split (str): Dataset split to load (default: "train")

        Returns:
            List[Document]: List of processed documents from the dataset

        Raises:
            ValueError: If unable to load the dataset
        """
        try:
            print(f"Loading {dataset_name} dataset from Hugging Face...")
            
            # Special handling for PDF-heavy datasets
            is_pdf_dataset = "medrax" in dataset_name.lower()
            batch_size = 20 if is_pdf_dataset else 100  # Smaller batches for PDF datasets
            
            # Load dataset
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
            
            # Configure text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
            )
            
            # Track progress
            successfully_processed = 0
            failed_items = 0
            timeout_failures = 0
            extraction_failures = 0
            failed_pdfs = []  # Track details of failed PDFs
            all_documents = []
            total_items = len(dataset)
            
            print(f"Processing {total_items} items in batches of {batch_size}...")
            print(f"PDF processing timeout set to 30 seconds per PDF")
            
            # Process in batches
            for batch_start in range(0, total_items, batch_size):
                batch_end = min(batch_start + batch_size, total_items)
                batch_num = (batch_start // batch_size) + 1
                total_batches = (total_items + batch_size - 1) // batch_size
                
                print(f"\nProcessing batch {batch_num}/{total_batches} (items {batch_start}-{batch_end-1})...")
                
                batch_documents = []
                batch_failed = 0
                
                # Process items in this batch
                for idx in range(batch_start, batch_end):
                    try:
                        item = dataset[idx]
                        content = None
                        item_metadata = {
                            "source": f"{dataset_name}",
                            "dataset": dataset_name,
                            "split": split,
                            "item_index": idx,
                        }
                        
                        # Extract identifying information for better tracking
                        item_identifier = f"Index {idx}"
                        if "title" in item:
                            item_identifier += f" - Title: {item['title']}"
                        elif "id" in item:
                            item_identifier += f" - ID: {item['id']}"
                        elif "name" in item:
                            item_identifier += f" - Name: {item['name']}"
                        elif "filename" in item:
                            item_identifier += f" - File: {item['filename']}"
                        
                        # Try to extract first few words from content as identifier
                        content_preview = None
                        if "content" in item:
                            content_preview = item["content"]
                        elif "text" in item:
                            content_preview = item["text"]
                        
                        if content_preview and len(str(content_preview).split()) > 0:
                            item_identifier += f" - Preview: {str(content_preview).split()[:5]}..."
                        
                        print(f"Processing item {item_identifier}...")
                        
                        # Handle different dataset structures
                        if "content" in item:
                            content = item["content"]
                        elif "text" in item:
                            content = item["text"]
                        elif "pdf" in item and item["pdf"] is not None:
                            # Use safe PDF extraction with timeout
                            pdf_data = item["pdf"]
                            
                            print(f"Processing PDF at index {idx}...")
                            content, pdf_metadata = safe_extract_pdf_text(pdf_data, timeout_seconds=180)
                            
                            # Try to get a content preview for identification
                            if content and len(content.strip()) > 0:
                                preview_words = content.strip().split()[:8]
                                content_preview = " ".join(preview_words) + "..." if len(preview_words) >= 8 else " ".join(preview_words)
                                item_identifier += f" - Content: {content_preview}"
                            
                            if content is None:
                                print(f"Failed to extract text from PDF at index {idx} - skipping")
                                failed_items += 1
                                batch_failed += 1
                                timeout_failures += 1
                                failed_pdfs.append({
                                    "index": idx,
                                    "identifier": item_identifier,
                                    "reason": "timeout",
                                    "details": "PDF processing timed out after 180 seconds"
                                })
                                continue
                            
                            if not content.strip():
                                print(f"Warning: No text extracted from PDF at index {idx} - skipping")
                                failed_items += 1
                                batch_failed += 1
                                extraction_failures += 1
                                failed_pdfs.append({
                                    "index": idx,
                                    "identifier": item_identifier,
                                    "reason": "no_text",
                                    "details": "No text could be extracted from PDF"
                                })
                                continue
                            
                            # Add PDF metadata
                            if pdf_metadata:
                                item_metadata.update(pdf_metadata)
                        else:
                            # Try to find text fields
                            text_fields = [k for k in item.keys() if isinstance(item[k], str) and len(str(item[k])) > 50]
                            if text_fields:
                                content = item[text_fields[0]]
                            else:
                                print(f"Warning: Could not find text content in item {idx}")
                                failed_items += 1
                                batch_failed += 1
                                extraction_failures += 1
                                failed_pdfs.append({
                                    "index": idx,
                                    "identifier": item_identifier,
                                    "reason": "no_content",
                                    "details": f"No text fields found in item keys: {list(item.keys())}"
                                })
                                continue

                        if not content:
                            failed_items += 1
                            batch_failed += 1
                            failed_pdfs.append({
                                "index": idx,
                                "identifier": item_identifier,
                                "reason": "empty_content",
                                "details": "Content was empty after processing"
                            })
                            continue
                            
                        # Add additional metadata
                        for key, value in item.items():
                            if key not in ["content", "text", "pdf"] and isinstance(value, (str, int, float, bool)):
                                item_metadata[key] = value

                        # Split long documents into chunks
                        if len(content) > self.config.chunk_size * 1.5:
                            temp_doc = Document(page_content=content, metadata=item_metadata)
                            split_docs = text_splitter.split_documents([temp_doc])
                            
                            for i, doc in enumerate(split_docs):
                                doc.metadata["chunk_index"] = i
                                doc.metadata["total_chunks"] = len(split_docs)
                            
                            batch_documents.extend(split_docs)
                        else:
                            doc = Document(
                                page_content=content,
                                metadata=item_metadata,
                            )
                            batch_documents.append(doc)

                        successfully_processed += 1
                        
                    except Exception as e:
                        print(f"Error processing item {idx}: {str(e)}")
                        failed_items += 1
                        batch_failed += 1
                        failed_pdfs.append({
                            "index": idx,
                            "identifier": item_identifier,
                            "reason": "exception",
                            "details": f"Exception during processing: {str(e)}"
                        })
                        continue
                
                # Add batch documents to all documents
                all_documents.extend(batch_documents)
                batch_success = len(range(batch_start, batch_end)) - batch_failed
                print(f"Batch {batch_num} complete: {len(batch_documents)} documents created from {batch_success} successful PDFs")
                if batch_failed > 0:
                    print(f"Batch {batch_num} failures: {batch_failed} PDFs failed to process")
                
                # Memory cleanup after each batch
                if is_pdf_dataset:
                    print("Performing memory cleanup...")
                    del batch_documents
                    gc.collect()
                    
                    # Give a brief pause for system cleanup
                    time.sleep(1)
                    
                    # Report memory usage
                    if hasattr(sys, 'getsizeof'):
                        try:
                            mem_mb = sys.getsizeof(all_documents) / 1024 / 1024
                            print(f"Current document memory usage: {mem_mb:.1f} MB")
                        except:
                            pass

            print(f"\n{'='*60}")
            print(f"Dataset processing complete!")
            print(f"{'='*60}")
            print(f"Total documents created: {len(all_documents)}")
            print(f"Successfully processed: {successfully_processed} items")
            print(f"Failed items: {failed_items}")
            print(f"  - Timeout failures: {timeout_failures}")
            print(f"  - Extraction failures: {extraction_failures}")
            print(f"  - Other failures: {failed_items - timeout_failures - extraction_failures}")
            print(f"Success rate: {successfully_processed/(successfully_processed + failed_items)*100:.1f}%")
            
            # Display details of failed PDFs
            if failed_pdfs:
                print(f"\n{'='*60}")
                print(f"FAILED PDF DETAILS:")
                print(f"{'='*60}")
                for i, failed_pdf in enumerate(failed_pdfs, 1):
                    print(f"{i}. {failed_pdf['identifier']}")
                    print(f"   Reason: {failed_pdf['reason'].upper()}")
                    print(f"   Details: {failed_pdf['details']}")
                    if i < len(failed_pdfs):
                        print()
            
            print(f"{'='*60}")
            
            return all_documents

        except Exception as e:
            print(f"Error loading {dataset_name}: {str(e)}")
            raise ValueError(f"Failed to load dataset {dataset_name}: {str(e)}")
