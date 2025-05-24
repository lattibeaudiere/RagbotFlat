import os
import pickle
import json
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
from walrus_client import WalrusStorageClient
from document_map import DocumentMap
from retriever import DocumentRetriever

class DocumentIngester:
    def __init__(self, data_dir: str = None, vector_store_dir: str = "vector_store", use_walrus: bool = True):
        if data_dir is None:
            # Use the data directory in the project root
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        self.data_dir = data_dir
        self.vector_store_dir = vector_store_dir
        self.vectorizer = TfidfVectorizer()
        self.documents: List[str] = []
        self.document_metadata: List[Dict] = []
        self.use_walrus = use_walrus
        
        # Initialize Walrus client if using Walrus
        if self.use_walrus:
            self.walrus_client = WalrusStorageClient()
            self.doc_map = DocumentMap()
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(vector_store_dir, exist_ok=True)

    def load_documents(self):
        """Load documents from the data directory and/or Walrus storage"""
        supported_extensions = {'.py', '.md', '.txt', '.json', '.sol'}
        processed_files = set()
        
        # First, try to load from Walrus if enabled
        if self.use_walrus:
            print("Checking for documents in Walrus storage...")
            for filename in self.doc_map.list_documents():
                if any(filename.endswith(ext) for ext in supported_extensions):
                    try:
                        blob_id = self.doc_map.get_blob_id(filename)
                        if blob_id:
                            content = self.walrus_client.retrieve_document(blob_id).decode('utf-8')
                            self.documents.append(content)
                            # Create a virtual file path for consistency
                            virtual_path = os.path.join(self.data_dir, filename)
                            self.document_metadata.append({
                                'file_path': virtual_path,
                                'file_name': filename,
                                'blob_id': blob_id,
                                'source': 'walrus'
                            })
                            processed_files.add(filename)
                            print(f"Loaded {filename} from Walrus storage")
                    except Exception as e:
                        print(f"Error loading {filename} from Walrus: {str(e)}")
        
        # Then load from local filesystem
        print("Loading documents from local filesystem...")
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file in processed_files:
                    # Skip files already loaded from Walrus
                    continue
                    
                if any(file.endswith(ext) for ext in supported_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            self.documents.append(content)
                            self.document_metadata.append({
                                'file_path': os.path.abspath(file_path),
                                'file_name': file,
                                'source': 'local'
                            })
                            
                            # If using Walrus, also store the document there
                            if self.use_walrus:
                                try:
                                    metadata = {
                                        "filename": file,
                                        "content_type": "text/plain",
                                        "source": "ingest_script"
                                    }
                                    blob_id = self.walrus_client.store_document(content, metadata)
                                    self.doc_map.add_document(file, blob_id, metadata)
                                    print(f"Stored {file} in Walrus storage with blob_id: {blob_id}")
                                except Exception as e:
                                    print(f"Error storing {file} in Walrus: {str(e)}")
                            
                            print(f"Loaded {file} from local filesystem")
                    except Exception as e:
                        print(f"Error loading {file_path}: {str(e)}")

    def create_embeddings(self):
        """Create TF-IDF embeddings and FAISS index"""
        if not self.documents:
            print("No documents loaded. Please load documents first.")
            return

        # Create TF-IDF embeddings
        tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        embeddings = tfidf_matrix.toarray().astype('float32')

        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Save the index and metadata locally
        faiss.write_index(index, os.path.join(self.vector_store_dir, "faiss_index.bin"))
        
        with open(os.path.join(self.vector_store_dir, "vectorizer.pkl"), 'wb') as f:
            pickle.dump(self.vectorizer, f)
            
        with open(os.path.join(self.vector_store_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump(self.document_metadata, f)
        
        # If using Walrus, also save to Walrus storage
        if self.use_walrus:
            try:
                # Initialize a retriever to use its save_artifacts_to_walrus method
                retriever = DocumentRetriever(self.vector_store_dir)
                retriever.index = index
                retriever.vectorizer = self.vectorizer
                retriever.document_metadata = self.document_metadata
                
                # Save artifacts to Walrus
                success = retriever.save_artifacts_to_walrus()
                if success:
                    print("Successfully saved vector store artifacts to Walrus storage")
                else:
                    print("Failed to save vector store artifacts to Walrus storage")
            except Exception as e:
                print(f"Error saving to Walrus: {str(e)}")

    def process_documents(self):
        """Process all documents and create embeddings"""
        print("Loading documents...")
        self.load_documents()
        print(f"Loaded {len(self.documents)} documents")
        
        print("Creating embeddings...")
        self.create_embeddings()
        print("Embeddings created and saved successfully")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ingest documents for RAG')
    parser.add_argument('--no-walrus', action='store_true', help='Disable Walrus storage')
    args = parser.parse_args()
    
    use_walrus = not args.no_walrus
    print(f"Using Walrus storage: {use_walrus}")
    
    ingester = DocumentIngester(use_walrus=use_walrus)
    ingester.process_documents() 