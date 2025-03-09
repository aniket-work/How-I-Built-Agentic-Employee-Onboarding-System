# ai_onboarding.py (Updated)
import os
import uuid
import nltk
import torch
from groq._utils import lru_cache
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader

nltk.download('punkt', quiet=True)
load_dotenv()


class OnboardingVectorDB:
    def __init__(self, collection: str = "hr_docs"):
        self.client = chromadb.PersistentClient(path="./onboarding_db")
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"}  # Better similarity metric
        )

    def ingest_document(self, file_path: str, chunk_size: int = 3):  # Reduced chunk size
        """Improved document processing with metadata"""
        text = self._extract_text(file_path)
        sentences = sent_tokenize(text)
        chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

        # Add document-type metadata
        doc_type = "hr" if "hr" in file_path.lower() else "technical"
        metadatas = [{"source": file_path, "type": doc_type} for _ in chunks]

        self.collection.add(
            documents=chunks,
            ids=[str(uuid.uuid4()) for _ in chunks],
            metadatas=metadatas
        )

    def _extract_text(self, file_path: str) -> str:
        """Enhanced PDF text extraction"""
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


class HRAssistant:
    def __init__(self):
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.db = OnboardingVectorDB()

    @lru_cache(maxsize=100)
    def answer_query(self, query: str, role: str = "general") -> str:
        """Improved RAG implementation"""
        # Enhanced retrieval with metadata filtering
        results = self.db.collection.query(
            query_texts=[query],
            n_results=3,
            where={"type": "hr"},  # Filter to HR documents
            include=["documents", "metadatas"]
        )

        # Fallback if no results found
        if not results['documents']:
            return "I couldn't find relevant information in our documents. Please consult HR directly."

        context = "\n\n".join(results['documents'][0])

        # Stronger prompt engineering
        response = self.groq.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{
                "role": "system",
                "content": f"""You are an HR assistant for Aniket AI Company. 
                Strictly use ONLY this context to answer:
                {context}
                If unsure, say 'Please consult HR documentation'."""
            }, {
                "role": "user",
                "content": query
            }],
            temperature=0.1,  # More deterministic
            top_p=0.3
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    assistant = HRAssistant()

    # Ingest documents with explicit types
    assistant.db.ingest_document("hr_policies.pdf")
    assistant.db.ingest_document("technical_handbook.pdf")

    # Test query
    print(assistant.answer_query(
        "What is the company's leave policy?",
        role="engineering"
    ))