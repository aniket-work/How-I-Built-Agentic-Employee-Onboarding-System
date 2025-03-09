# src/vector_db.py
"""
Vector database functionality for the AI Onboarding System.
"""

import uuid
import chromadb
from chromadb.utils import embedding_functions
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader
from typing import List, Dict, Any

from src.utils import load_config


class OnboardingVectorDB:
    """Vector database for storing and retrieving onboarding documents."""

    def __init__(self):
        """Initialize the vector database with configuration settings."""
        config = load_config()
        db_config = config['database']

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_config['path'])

        # Initialize embedding function
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=db_config['embedding_model']
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=db_config['collection'],
            embedding_function=self.embedder,
            metadata={"hnsw:space": db_config['similarity_metric']}
        )

    def ingest_document(self, file_path: str, chunk_size: int = 3) -> None:
        """
        Process documents into vector database.

        Args:
            file_path: Path to the document file
            chunk_size: Number of sentences per chunk
        """
        text = self._extract_text(file_path)
        sentences = sent_tokenize(text)
        chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

        # Determine document type based on filename
        doc_type = "hr" if "hr" in file_path.lower() else "technical"
        metadatas = [{"source": file_path, "type": doc_type} for _ in chunks]

        # Add chunks to vector database
        self.collection.add(
            documents=chunks,
            ids=[str(uuid.uuid4()) for _ in chunks],
            metadatas=metadatas
        )

        return len(chunks)

    def _extract_text(self, file_path: str) -> str:
        """
        Extract text from various file formats.

        Args:
            file_path: Path to the document file

        Returns:
            Extracted text content
        """
        if file_path.endswith('.pdf'):
            text = ""
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        else:
            with open(file_path, 'r') as f:
                return f.read()

    def query_documents(self, query_text: str, doc_type: str = None, n_results: int = 3) -> Dict[str, Any]:
        """
        Query the vector database for relevant documents.

        Args:
            query_text: The query text
            doc_type: Type of document to filter by (optional)
            n_results: Number of results to return

        Returns:
            Query results
        """
        where_filter = {"type": doc_type} if doc_type else None

        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas"]
        )