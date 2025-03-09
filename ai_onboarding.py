# ai_onboarding.py
import os
import uuid
import nltk
import torch
from functools import lru_cache
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
from datetime import datetime
import textwrap
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)
nltk.download('punkt', quiet=True)
load_dotenv()

# Color scheme
COLORS = {
    "title": Fore.CYAN + Style.BRIGHT,
    "input": Fore.YELLOW,
    "warning": Fore.RED,
    "success": Fore.GREEN,
    "header": Fore.MAGENTA + Style.BRIGHT,
    "border": Fore.BLUE,
    "text": Fore.WHITE
}


def format_section(title: str, content: list, color=COLORS["border"]):
    """Create consistent left-aligned bordered sections"""
    max_length = max(len(line) for line in content + [title]) if content else len(title)
    border_top = f"{color}╭{'─' * (max_length + 2)}╮"
    title_line = f"{color}│ {Style.BRIGHT}{title.upper().center(max_length)}{Style.NORMAL} {color}│"
    content_lines = [f"{color}│{COLORS['text']}  {line.ljust(max_length)}  {color}│" for line in content]
    border_bottom = f"{color}╰{'─' * (max_length + 2)}╯"
    return "\n".join([border_top, title_line] + content_lines + [border_bottom])


class OnboardingVectorDB:
    def __init__(self, collection: str = "hr_docs"):
        self.client = chromadb.PersistentClient(path="./onboarding_db")
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_document(self, file_path: str, chunk_size: int = 3):
        """Process documents into vector database"""
        text = self._extract_text(file_path)
        sentences = sent_tokenize(text)
        chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

        doc_type = "hr" if "hr" in file_path.lower() else "technical"
        metadatas = [{"source": file_path, "type": doc_type} for _ in chunks]

        self.collection.add(
            documents=chunks,
            ids=[str(uuid.uuid4()) for _ in chunks],
            metadatas=metadatas
        )

    def _extract_text(self, file_path: str) -> str:
        """Extract text from PDF files"""
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


class OnboardingAgent:
    def __init__(self):
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.db = OnboardingVectorDB()
        self.interaction_history = []
        self.user_context = {}

    def start_session(self):
        """Initialize onboarding session"""
        print(format_section("Welcome to Aniket AI Onboarding System", [], COLORS["title"]))
        self._collect_initial_info()
        self._show_help()
        self._main_interaction_loop()

    def _collect_initial_info(self):
        """Collect user information"""
        print(format_section("Let's Get Started", [], COLORS["success"]))
        self.user_context['name'] = input(f"{COLORS['input']}Your full name: ").strip()
        self.user_context['role'] = input(f"{COLORS['input']}Your job role: ").strip()
        self.user_context['start_date'] = datetime.now().strftime("%Y-%m-%d")
        print(f"\n{COLORS['success']}Welcome, {self.user_context['name']}! Setting up your onboarding...\n")

    def _main_interaction_loop(self):
        """Handle user commands"""
        while True:
            try:
                user_input = input(f"{COLORS['border']}Onboarding Assistant> {COLORS['text']}").strip().lower()
                if not user_input:
                    continue

                if user_input.startswith("ask "):
                    self._handle_question(user_input[4:].strip())
                elif user_input == "checklist":
                    self._show_checklist()
                elif user_input == "resources":
                    self._show_resources()
                elif user_input == "schedule":
                    self._show_schedule()
                elif user_input == "email":
                    self._generate_welcome_email()
                elif user_input == "help":
                    self._show_help()
                elif user_input in ["exit", "quit"]:
                    print(format_section("Thank You", ["Goodbye!"], COLORS["title"]))
                    break
                else:
                    print(format_section("Error", ["Unknown command. Type 'help' for options."], COLORS["warning"]))

            except KeyboardInterrupt:
                print(format_section("Session Interrupted", ["Exiting..."], COLORS["warning"]))
                break

    def _show_checklist(self):
        """Display onboarding progress"""
        checklist = [
            f"{COLORS['success']}✓ Complete HR paperwork",
            f"{COLORS['success']}✓ Set up company email",
            f"{COLORS['warning']}◻ Attend orientation session",
            f"{COLORS['warning']}◻ Complete security training",
            f"{COLORS['warning']}◻ Meet with team lead"
        ]
        progress = f"{COLORS['header']}Progress: {COLORS['success']}2/5 tasks completed"
        print(format_section("Onboarding Checklist", checklist + [progress]))

    def _show_schedule(self):
        """Display training schedule"""
        schedule = [
            f"{COLORS['header']}Day 1 ({self.user_context['start_date']}):",
            f"{COLORS['success']}  9:00 AM - Welcome Breakfast",
            f"{COLORS['border']} 10:00 AM - HR Orientation",
            f"{COLORS['text']}  2:00 PM - Workstation Setup",
            "",
            f"{COLORS['header']}Day 2:",
            f"{COLORS['success']}  9:30 AM - Team Introduction",
            f"{COLORS['border']} 11:00 AM - Systems Training",
            "",
            f"{COLORS['header']}Day 3:",
            f"{COLORS['warning']} 10:00 AM - Security Briefing",
            f"{COLORS['success']}  1:00 PM - Role-Specific Training",
            f"{COLORS['border']}Full calendar: {COLORS['text']}https://calendar.aniket-ai.com"
        ]
        print(format_section("Onboarding Schedule", schedule))

    def _show_resources(self):
        """Show learning resources"""
        role = self.user_context.get('role', 'general').lower()
        resources = [
            f"{COLORS['success']}1. Employee Portal: {COLORS['text']}https://portal.aniket-ai.com",
            f"{COLORS['success']}2. HR Policies: {COLORS['text']}https://hr.aniket-ai.com/policies",
            f"{COLORS['success']}3. IT Support: {COLORS['text']}helpdesk@aniket-ai.com",
            f"{COLORS['success']}4. Learning Platform: {COLORS['text']}https://learn.aniket-ai.com"
        ]
        if 'engineer' in role:
            resources.extend([
                f"{COLORS['success']}5. Code Repository: {COLORS['text']}https://git.aniket-ai.com",
                f"{COLORS['success']}6. ML Training: {COLORS['text']}https://learn.aniket-ai.com/ml-101"
            ])
        print(format_section("Learning Resources", resources))

    def _generate_welcome_email(self):
        """Generate welcome email"""
        email_content = [
            f"{COLORS['header']}Subject: Welcome to Aniket AI, {self.user_context['name']}!",
            "",
            f"{COLORS['text']}Dear {self.user_context['name']},",
            "",
            f"{COLORS['success']}We're excited to have you join us as a {self.user_context['role']}!",
            "",
            f"{COLORS['border']}Your onboarding schedule:",
            f"{COLORS['text']}- First day: {self.user_context['start_date']}",
            f"{COLORS['text']}- Team meeting: {self.user_context['start_date']} 10:00 AM",
            f"{COLORS['text']}- Equipment setup: IT Department (Floor 3)",
            "",
            f"{COLORS['header']}Best regards,",
            f"{COLORS['success']}Aniket AI HR Team"
        ]
        print(format_section("Welcome Email", email_content, COLORS["success"]))

    def _show_help(self):
        """Show help menu"""
        help_content = [
            f"{COLORS['success']}ask [question]  : Get policy answers",
            f"{COLORS['border']}checklist       : View progress",
            f"{COLORS['success']}resources       : Learning materials",
            f"{COLORS['border']}schedule        : Training timeline",
            f"{COLORS['success']}email           : Generate welcome email",
            f"{COLORS['border']}help            : Show this menu",
            f"{COLORS['warning']}exit            : End session"
        ]
        print(format_section("Available Commands", help_content))

    @lru_cache(maxsize=100)
    def _handle_question(self, question: str):
        """Answer questions using RAG"""
        results = self.db.collection.query(
            query_texts=[question],
            n_results=3,
            where={"type": "hr"},
            include=["documents"]
        )

        context = "\n".join(results['documents'][0]) if results['documents'] else ""
        response = self.groq.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{
                "role": "system",
                "content": f"""Answer as HR assistant for {self.user_context['name']}.
                Context: {context}
                Be concise and professional."""
            }, {
                "role": "user",
                "content": question
            }],
            temperature=0.3
        )
        answer = response.choices[0].message.content
        print(format_section("Answer", [answer], COLORS["success"]))


if __name__ == "__main__":
    agent = OnboardingAgent()
    agent.start_session()