import requests
import base64
import json
import os
from datetime import datetime
from dotenv import load_dotenv

class WalrusStorageClient:
    def __init__(self, api_url=None, api_key=None):
        load_dotenv()
        self.api_url = api_url or os.getenv("WALRUS_API_URL", "http://146.148.47.40")
        self.api_key = api_key or os.getenv("WALRUS_API_KEY", "ce4af219-22ab-42d7-9c5d-2e4dc176a4f4")
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
    
    def store_document(self, file_content, metadata=None):
        """Store a document in Walrus storage
        
        Args:
            file_content (bytes or str): Content to store. If string, it will be encoded to bytes
            metadata (dict, optional): Metadata to associate with the document
            
        Returns:
            str: The blob_id of the stored document
        """
        if metadata is None:
            metadata = {}
        
        # Ensure file_content is bytes
        if isinstance(file_content, str):
            file_content = file_content.encode('utf-8')
            
        encoded_content = base64.b64encode(file_content).decode('utf-8')
        
        payload = {
            "data": encoded_content,
            "metadata": metadata
        }
        
        response = requests.post(
            f"{self.api_url}/api/v1/store",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()["blob_id"]
        else:
            raise Exception(f"Failed to store document: {response.text}")
    
    def retrieve_document(self, blob_id):
        """Retrieve a document from Walrus storage
        
        Args:
            blob_id (str): The blob_id of the document to retrieve
            
        Returns:
            bytes: The document content as bytes
        """
        response = requests.get(
            f"{self.api_url}/api/v1/retrieve/{blob_id}",
            headers=self.headers
        )
        
        if response.status_code == 200:
            encoded_data = response.json()["data"]
            return base64.b64decode(encoded_data)
        else:
            raise Exception(f"Failed to retrieve document: {response.text}")
    
    def list_documents(self):
        """List all documents in Walrus storage
        
        Returns:
            list: List of blob metadata
        """
        response = requests.get(
            f"{self.api_url}/api/v1/list",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()["blobs"]
        else:
            raise Exception(f"Failed to list documents: {response.text}")
            
    def get_blob_info(self, blob_id):
        """Get information about a blob
        
        Args:
            blob_id (str): The blob_id to get information about
            
        Returns:
            dict: Information about the blob
        """
        # First try to get the blob metadata from the list endpoint
        try:
            blobs = self.list_documents()
            for blob in blobs:
                if blob.get('id') == blob_id:
                    return {
                        'blob_id': blob_id,
                        'metadata': blob.get('metadata', {}),
                        'timestamp': blob.get('timestamp'),
                        'size': blob.get('size'),
                        'blockchain_info': 'Available in Sui wallet transactions'
                    }
        except Exception as e:
            print(f"Error getting blob list: {str(e)}")
            
        # If we couldn't find it in the list, try to retrieve it to at least confirm it exists
        try:
            # Just check if we can retrieve it
            self.retrieve_document(blob_id)
            return {
                'blob_id': blob_id,
                'exists': True,
                'message': 'Blob exists but detailed metadata is not available',
                'blockchain_info': 'Available in Sui wallet transactions'
            }
        except Exception as e:
            raise Exception(f"Failed to get blob info: {str(e)}")

# Example usage
if __name__ == "__main__":
    client = WalrusStorageClient()
    
    # Test storing a document
    try:
        blob_id = client.store_document(
            "This is a test document",
            {"filename": "test.txt", "content_type": "text/plain"}
        )
        print(f"Stored document with blob_id: {blob_id}")
        
        # Test retrieving the document
        content = client.retrieve_document(blob_id)
        print(f"Retrieved document content: {content.decode('utf-8')}")
        
        # Test listing documents
        blobs = client.list_documents()
        print(f"Found {len(blobs)} blobs")
    except Exception as e:
        print(f"Error: {e}")
