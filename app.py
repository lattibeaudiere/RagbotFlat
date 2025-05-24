from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from datetime import datetime

# Ensure the backend can import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_chatbot import RAGChatbot
from chatbot import ChutesChatbot
from walrus_client import WalrusStorageClient
from document_map import DocumentMap
import asyncio

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://sewnrag.netlify.app", "http://localhost:3000"], "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}})
# Enable CORS for all routes and origins

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize Walrus client
walrus_client = WalrusStorageClient()
print(f"Initialized Walrus client with API URL: {walrus_client.api_url}")
doc_map = DocumentMap()
print("Initialized Document Map")

# Test Walrus connection
try:
    blobs = walrus_client.list_documents()
    print(f"Successfully connected to Walrus API. Found {len(blobs)} blobs.")
except Exception as e:
    print(f"ERROR connecting to Walrus API: {str(e)}")
    print(f"Walrus API URL: {walrus_client.api_url}")
    print(f"Walrus API Key: {walrus_client.api_key[:5]}...{walrus_client.api_key[-5:] if len(walrus_client.api_key) > 10 else ''}")
    print("Continuing with local storage only")

# Load both chatbot systems once
rag_bot = RAGChatbot()
llm_bot = ChutesChatbot()

@app.route('/chat/rag', methods=['POST'])
def chat_rag():
    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    response = asyncio.run(rag_bot.get_response(user_message))
    return jsonify({'response': response})

@app.route('/chat/llm', methods=['POST'])
def chat_llm():
    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    response = asyncio.run(llm_bot.get_response(user_message))
    return jsonify({'response': response})

@app.route('/chat/blended', methods=['POST'])
def chat_blended():
    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    # Retrieve RAG context
    relevant_docs = rag_bot.retriever.retrieve_documents(user_message)
    context = rag_bot._format_context(relevant_docs)
    # Compose blended prompt
    blended_prompt = (
        "You are a helpful assistant. Use the following context if relevant, but also use your own knowledge if needed.\n\n" +
        context +
        f"User: {user_message}"
    )
    # Use LLM to answer
    response = asyncio.run(llm_bot.get_response(blended_prompt))
    return jsonify({'response': response})

@app.route('/files', methods=['GET'])
def list_txt_files():
    # Get files from Walrus
    walrus_files = []
    try:
        walrus_files = doc_map.list_documents()
        print(f"Found {len(walrus_files)} files in Walrus storage")
    except Exception as e:
        print(f"Error listing files from Walrus: {str(e)}")
    
    # Get files from local directory
    local_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
    print(f"Found {len(local_files)} files in local storage")
    
    # Combine and deduplicate
    all_files = list(set(walrus_files + local_files))
    
    return jsonify({'files': all_files, 'walrus_count': len(walrus_files), 'local_count': len(local_files)})

@app.route('/files/upload', methods=['POST'])
def upload_txt_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.txt'):
        return jsonify({'error': 'Invalid file name'}), 400
    
    # Read file content
    file_content = file.read()
    
    try:
        # Store in Walrus
        print(f"Attempting to store {file.filename} in Walrus storage...")
        metadata = {
            "filename": file.filename,
            "content_type": "text/plain",
            "timestamp": datetime.now().isoformat(),
            "source": "web_upload"
        }
        print(f"Created metadata: {metadata}")
        try:
            blob_id = walrus_client.store_document(file_content, metadata)
            print(f"Successfully stored in Walrus with blob_id: {blob_id}")
            doc_map.add_document(file.filename, blob_id, metadata)
            print(f"Added to document map: {file.filename} -> {blob_id}")
        except Exception as walrus_e:
            print(f"ERROR storing in Walrus: {str(walrus_e)}")
            raise walrus_e
        
        # Also save locally for backward compatibility
        save_path = os.path.join(DATA_DIR, file.filename)
        print(f"Saving uploaded file to: {save_path}")
        with open(save_path, 'wb') as f:
            f.write(file_content)
        
        # Trigger ingestion
        os.system(f'python {os.path.join(os.path.dirname(__file__), "ingest.py")}') 
        rag_bot.retriever.load_artifacts()  # Reload vector store after ingestion
        
        return jsonify({
            'success': True, 
            'filename': file.filename,
            'blob_id': blob_id,
            'storage': 'walrus'
        })
    except Exception as e:
        # Fall back to local storage only
        try:
            save_path = os.path.join(DATA_DIR, file.filename)
            print(f"Saving uploaded file to local storage only: {save_path}")
            with open(save_path, 'wb') as f:
                f.write(file_content)
            
            # Trigger ingestion
            os.system(f'python {os.path.join(os.path.dirname(__file__), "ingest.py")} --no-walrus') 
            rag_bot.retriever.load_artifacts()  # Reload vector store after ingestion
            
            return jsonify({
                'success': True, 
                'filename': file.filename,
                'storage': 'local',
                'walrus_error': str(e)
            })
        except Exception as local_e:
            return jsonify({'error': f'Failed to save file: {str(local_e)}'}), 500

@app.route('/files/append', methods=['POST'])
def append_to_txt_file():
    data = request.json
    filename = data.get('filename')
    content = data.get('content')
    if not filename or not content:
        return jsonify({'error': 'Missing filename or content'}), 400
    
    try:
        # Check if file exists in Walrus
        blob_id = doc_map.get_blob_id(filename)
        if blob_id:
            # Get existing content from Walrus
            existing_content = walrus_client.retrieve_document(blob_id).decode('utf-8')
            
            # Append new content
            new_content = existing_content + '\n' + content
            
            # Store updated content in Walrus
            metadata = {
                "filename": filename,
                "content_type": "text/plain",
                "timestamp": datetime.now().isoformat(),
                "updated": True,
                "source": "web_append"
            }
            new_blob_id = walrus_client.store_document(new_content.encode('utf-8'), metadata)
            doc_map.add_document(filename, new_blob_id, metadata)
            print(f"Updated {filename} in Walrus storage with new blob_id: {new_blob_id}")
            
            # Also update local file for backward compatibility
            file_path = os.path.join(DATA_DIR, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        else:
            # Check if file exists locally
            file_path = os.path.join(DATA_DIR, filename)
            if not os.path.exists(file_path):
                return jsonify({'error': 'File does not exist in Walrus or locally'}), 404
            
            # Read existing content
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
            
            # Append new content
            new_content = existing_content + '\n' + content
            
            # Store in Walrus
            metadata = {
                "filename": filename,
                "content_type": "text/plain",
                "timestamp": datetime.now().isoformat(),
                "source": "web_append"
            }
            blob_id = walrus_client.store_document(new_content.encode('utf-8'), metadata)
            doc_map.add_document(filename, blob_id, metadata)
            print(f"Stored {filename} in Walrus storage with blob_id: {blob_id}")
            
            # Update local file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        
        # Trigger ingestion
        os.system(f'python {os.path.join(os.path.dirname(__file__), "ingest.py")}') 
        rag_bot.retriever.load_artifacts()  # Reload vector store after ingestion
        
        return jsonify({
            'success': True, 
            'filename': filename,
            'storage': 'walrus'
        })
    except Exception as e:
        # Fall back to local storage only
        try:
            file_path = os.path.join(DATA_DIR, filename)
            if not os.path.exists(file_path):
                return jsonify({'error': 'File does not exist locally'}), 404
                
            # Append to local file
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write('\n' + content)
            
            # Trigger ingestion
            os.system(f'python {os.path.join(os.path.dirname(__file__), "ingest.py")} --no-walrus') 
            rag_bot.retriever.load_artifacts()  # Reload vector store after ingestion
            
            return jsonify({
                'success': True, 
                'filename': filename,
                'storage': 'local',
                'walrus_error': str(e)
            })
        except Exception as local_e:
            return jsonify({'error': f'Failed to append to file: {str(local_e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    # Check Walrus connection
    walrus_status = "ok"
    try:
        walrus_client.list_documents()
    except Exception as e:
        walrus_status = f"error: {str(e)}"
    
    return jsonify({
        'status': 'ok',
        'walrus_status': walrus_status,
        'storage_type': 'walrus' if walrus_status == "ok" else 'local'
    })

@app.route('/blob/info/<blob_id>', methods=['GET'])
def get_blob_info(blob_id):
    """Get blockchain information about a blob"""
    try:
        # Use the WalrusStorageClient to get blob info
        blob_info = walrus_client.get_blob_info(blob_id)
        return jsonify(blob_info)
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
