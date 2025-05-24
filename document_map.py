import json
import os
from datetime import datetime

class DocumentMap:
    def __init__(self, map_file="document_map.json"):
        self.map_file = map_file
        self.document_map = self._load_map()
    
    def _load_map(self):
        """Load the document map from disk or create a new one"""
        try:
            with open(self.map_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def save_map(self):
        """Save the document map to disk"""
        with open(self.map_file, "w") as f:
            json.dump(self.document_map, f)
    
    def add_document(self, filename, blob_id, metadata=None):
        """Add a document to the map"""
        if metadata is None:
            metadata = {}
        
        self.document_map[filename] = {
            "blob_id": blob_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        }
        self.save_map()
    
    def get_blob_id(self, filename):
        """Get the blob_id for a filename"""
        if filename not in self.document_map:
            return None
        return self.document_map[filename]["blob_id"]
    
    def list_documents(self):
        """List all documents in the map"""
        return list(self.document_map.keys())
    
    def document_exists(self, filename):
        """Check if a document exists in the map"""
        return filename in self.document_map
    
    def remove_document(self, filename):
        """Remove a document from the map"""
        if filename in self.document_map:
            del self.document_map[filename]
            self.save_map()
            return True
        return False
