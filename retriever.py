import os
import pickle
import faiss
import json
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from walrus_client import WalrusStorageClient
from document_map import DocumentMap

class DocumentRetriever:
    def __init__(self, vector_store_dir: str = "vector_store"):
        self.vector_store_dir = vector_store_dir
        self.index = None
        self.vectorizer = None
        self.document_metadata = None
        
        # Initialize Walrus client
        self.walrus_client = WalrusStorageClient()
        
        # Initialize document map
        self.doc_map = DocumentMap()
        
        # Vector store blob IDs
        self.vector_store_blob_ids_file = "vector_store_blob_ids.json"
        self.vector_store_blob_ids = self._load_vector_store_blob_ids()
        
        # Load artifacts
        self.load_artifacts()

    def _load_vector_store_blob_ids(self):
        """Load vector store blob IDs from disk"""
        try:
            if os.path.exists(self.vector_store_blob_ids_file):
                with open(self.vector_store_blob_ids_file, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            print(f"Error loading vector store blob IDs: {str(e)}")
            return {}
    
    def _save_vector_store_blob_ids(self):
        """Save vector store blob IDs to disk"""
        with open(self.vector_store_blob_ids_file, 'w') as f:
            json.dump(self.vector_store_blob_ids, f)
    
    def load_artifacts(self):
        """Load the FAISS index, vectorizer, and metadata from Walrus or local storage"""
        try:
            # First try to load from Walrus
            if self.vector_store_blob_ids and all(k in self.vector_store_blob_ids for k in ["index", "vectorizer", "metadata"]):
                try:
                    print("Attempting to load artifacts from Walrus...")
                    
                    # Load FAISS index
                    index_data = self.walrus_client.retrieve_document(self.vector_store_blob_ids["index"])
                    # Write to temporary file and load with read_index
                    temp_index_path = os.path.join(self.vector_store_dir, "temp_faiss_index.bin")
                    with open(temp_index_path, 'wb') as f:
                        f.write(index_data)
                    self.index = faiss.read_index(temp_index_path)
                    print("Loaded FAISS index from Walrus")
                    
                    # Load vectorizer
                    vectorizer_data = self.walrus_client.retrieve_document(self.vector_store_blob_ids["vectorizer"])
                    self.vectorizer = pickle.loads(vectorizer_data)
                    print("Loaded vectorizer from Walrus")
                    
                    # Load metadata
                    metadata_data = self.walrus_client.retrieve_document(self.vector_store_blob_ids["metadata"])
                    self.document_metadata = pickle.loads(metadata_data)
                    print("Loaded document metadata from Walrus")
                    
                    return
                except Exception as e:
                    print(f"Error loading artifacts from Walrus: {str(e)}")
                    print("Falling back to local storage")
            
            # Fall back to local storage
            print("Loading artifacts from local storage...")
            
            # Load FAISS index
            index_path = os.path.join(self.vector_store_dir, "faiss_index.bin")
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                print("Loaded FAISS index from local storage")
            else:
                print("FAISS index not found. Retrieval will be unavailable until files are ingested.")
                self.index = None

            # Load vectorizer
            vectorizer_path = os.path.join(self.vector_store_dir, "vectorizer.pkl")
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                print("Loaded vectorizer from local storage")
            else:
                self.vectorizer = None

            # Load metadata
            metadata_path = os.path.join(self.vector_store_dir, "metadata.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.document_metadata = pickle.load(f)
                print("Loaded document metadata from local storage")
            else:
                self.document_metadata = None

        except Exception as e:
            print(f"Error loading artifacts: {str(e)}")
            self.index = None
            self.vectorizer = None
            self.document_metadata = None

    def save_artifacts_to_walrus(self):
        """Save vector store artifacts to Walrus"""
        if not self.index or not self.vectorizer or not self.document_metadata:
            print("Vector store not initialized. Nothing to save.")
            return
        
        try:
            print("Saving vector store artifacts to Walrus...")
            
            # Save FAISS index
            index_data = faiss.serialize_index(self.index)
            index_blob_id = self.walrus_client.store_document(
                index_data,
                {"content_type": "application/octet-stream", "type": "faiss_index"}
            )
            self.vector_store_blob_ids["index"] = index_blob_id
            print(f"Saved FAISS index to Walrus with blob_id: {index_blob_id}")
            
            # Save vectorizer
            vectorizer_data = pickle.dumps(self.vectorizer)
            vectorizer_blob_id = self.walrus_client.store_document(
                vectorizer_data,
                {"content_type": "application/octet-stream", "type": "vectorizer"}
            )
            self.vector_store_blob_ids["vectorizer"] = vectorizer_blob_id
            print(f"Saved vectorizer to Walrus with blob_id: {vectorizer_blob_id}")
            
            # Save metadata
            metadata_data = pickle.dumps(self.document_metadata)
            metadata_blob_id = self.walrus_client.store_document(
                metadata_data,
                {"content_type": "application/octet-stream", "type": "document_metadata"}
            )
            self.vector_store_blob_ids["metadata"] = metadata_blob_id
            print(f"Saved document metadata to Walrus with blob_id: {metadata_blob_id}")
            
            # Save blob IDs
            self._save_vector_store_blob_ids()
            print("Saved vector store blob IDs to disk")
            
            return True
        except Exception as e:
            print(f"Error saving artifacts to Walrus: {str(e)}")
            return False
    
    def retrieve_documents(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve the k most relevant documents for a given query
        Returns a list of dictionaries containing document content and metadata
        """
        if not self.index or not self.vectorizer or not self.document_metadata:
            print("Vector store not initialized. No documents available for retrieval.")
            return []

        # Convert query to embedding
        query_embedding = self.vectorizer.transform([query]).toarray().astype('float32')

        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)

        # Get relevant documents
        relevant_docs = []
        for idx in indices[0]:
            if idx < len(self.document_metadata):
                doc_info = self.document_metadata[idx]
                try:
                    file_name = doc_info['file_name']
                    
                    # Try to get content from Walrus first
                    blob_id = self.doc_map.get_blob_id(file_name)
                    if blob_id:
                        try:
                            content = self.walrus_client.retrieve_document(blob_id).decode('utf-8')
                            relevant_docs.append({
                                'content': content,
                                'file_name': file_name,
                                'file_path': doc_info['file_path'],  # Keep for compatibility
                                'source': 'walrus'
                            })
                            continue
                        except Exception as e:
                            print(f"Error retrieving document from Walrus: {str(e)}")
                            # Fall back to local file
                    
                    # Fallback to local file
                    file_path = doc_info['file_path']
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            relevant_docs.append({
                                'content': content,
                                'file_name': file_name,
                                'file_path': file_path,
                                'source': 'local'
                            })
                except Exception as e:
                    print(f"Error reading document {doc_info['file_name']}: {str(e)}")

        return relevant_docs

if __name__ == "__main__":
    # Example usage
    retriever = DocumentRetriever()
    query = "example query"
    results = retriever.retrieve_documents(query)
    for doc in results:
        print(f"File: {doc['file_name']}")
        print(f"Content: {doc['content'][:200]}...")  # Print first 200 chars
        print("---") 