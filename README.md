# RAGBot with Blockchain Storage

A RAG (Retrieval-Augmented Generation) chatbot with blockchain-based storage using the Walrus protocol.

## Features

- Document ingestion and embedding generation
- Vector-based document retrieval
- Blockchain storage integration via Walrus protocol
- RESTful API for document management and chat

## Environment Variables

The application requires the following environment variables:

```
WALRUS_API_URL=http://146.148.47.40
WALRUS_API_KEY=ce4af219-22ab-42d7-9c5d-2e4dc176a4f4
CHUTES_API_TOKEN=your_token_here
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RagbotFlat.git
cd RagbotFlat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
gunicorn app:app
```

## API Endpoints

- `/files/upload` - Upload text files to the system
- `/files/list` - List all uploaded files
- `/chat` - Chat with the RAG system
- `/ingest` - Ingest documents into the vector store
- `/blob/info/<blob_id>` - Get information about a blob stored in the blockchain

## Blockchain Storage

This application uses the Walrus protocol for blockchain-based document storage, ensuring:
- Immutable document storage
- Decentralized access
- Verifiable document integrity

Blockchain information for stored documents can be viewed through the Sui wallet transactions.
