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
            metadata={"hnsw:space": "cosine"}
        )

    # Existing methods remain the same...


class OnboardingAgent:
    def __init__(self):
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.db = OnboardingVectorDB()
        self.interaction_history = []
        self.user_context = {}

    def start_session(self):
        """Initialize a new onboarding session"""
        print("\n" + "=" * 40)
        print("  Welcome to Aniket AI Onboarding System  ")
        print("=" * 40 + "\n")
        self._collect_initial_info()
        self._show_help()
        self._main_interaction_loop()

    def _collect_initial_info(self):
        """Collect basic user information"""
        print("Let's get started with your onboarding!\n")
        self.user_context['name'] = input("Your full name: ").strip()
        self.user_context['role'] = input("Your job role: ").strip()
        self.user_context['start_date'] = datetime.now().strftime("%Y-%m-%d")
        print("\nWelcome, {}! Setting up your onboarding...\n".format(self.user_context['name']))

    def _show_help(self):
        """Display available commands"""
        help_text = """
        Available commands:
        - ask [question]  : Get answers about company policies
        - checklist       : View onboarding progress
        - resources       : Show learning resources
        - schedule        : View training schedule
        - email           : Generate welcome email
        - help            : Show this help menu
        - exit            : End the session
        """
        print(textwrap.dedent(help_text))

    def _main_interaction_loop(self):
        """Handle continuous user interaction"""
        while True:
            try:
                user_input = input("\nOnboarding Assistant> ").strip().lower()
                if not user_input:
                    continue

                if user_input.startswith("ask "):
                    question = user_input[4:].strip()
                    self._handle_question(question)
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
                    print("\nThank you for using Aniket AI Onboarding System!")
                    break
                else:
                    print("Unknown command. Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\nSession interrupted. Exiting...")
                break

    def _show_resources(self):
        """Display role-specific learning resources"""
        role = self.user_context.get('role', 'general').lower()

        resources = {
            'engineering': [
                "1. Engineering Handbook: https://eng.aniket-ai.com",
                "2. Code Repository: https://git.aniket-ai.com",
                "3. ML Training Course: https://learn.aniket-ai.com/ml-101",
                "4. Infrastructure Docs: https://docs.aniket-ai.com/cloud"
            ],
            'general': [
                "1. Employee Portal: https://portal.aniket-ai.com",
                "2. HR Policies: https://hr.aniket-ai.com/policies",
                "3. IT Support: helpdesk@aniket-ai.com",
                "4. Learning Platform: https://learn.aniket-ai.com"
            ]
        }

        print("\nRecommended Resources:")
        for item in resources.get(role, resources['general']):
            print(f"  {item}")

    def _show_schedule(self):
        """Generate personalized onboarding schedule"""
        schedule = [
            f"Day 1 ({self.user_context['start_date']}):",
            "  9:00 AM - Welcome Breakfast",
            " 10:00 AM - HR Orientation",
            "  2:00 PM - Workstation Setup",
            "",
            "Day 2:",
            "  9:30 AM - Team Introduction",
            " 11:00 AM - Systems Training",
            "",
            "Day 3:",
            " 10:00 AM - Security Briefing",
            "  1:00 PM - Role-Specific Training"
        ]

        print("\nOnboarding Schedule:")
        print("\n".join(schedule))
        print("\nFull calendar available at: https://calendar.aniket-ai.com")

    @lru_cache(maxsize=100)
    def _handle_question(self, question: str):
        """Enhanced QA handling with context"""
        # Retrieve relevant information
        results = self.db.collection.query(
            query_texts=[question],
            n_results=3,
            include=["documents", "metadatas"]
        )

        # Prepare context-aware response
        context = "\n\n".join(results['documents'][0]) if results['documents'] else ""
        response = self.groq.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{
                "role": "system",
                "content": f"""You are an onboarding assistant for {self.user_context['name']}, 
                a new {self.user_context['role']} at Aniket AI. Use this context:
                {context}
                Respond professionally but conversationally."""
            }, {
                "role": "user",
                "content": question
            }],
            temperature=0.3,
            top_p=0.7
        )

        answer = response.choices[0].message.content
        self._format_response(answer)
        self._log_interaction(question, answer)

    def _format_response(self, text: str):
        """Format responses professionally"""
        wrapper = textwrap.TextWrapper(width=80, subsequent_indent='  ')
        print("\n" + "\n".join(wrapper.wrap(text)) + "\n")

    def _show_checklist(self):
        """Generate dynamic onboarding checklist"""
        checklist = [
            "✓ Complete HR paperwork",
            "✓ Set up company email",
            "◻ Attend orientation session",
            "◻ Complete security training",
            "◻ Meet with team lead"
        ]
        print("\nOnboarding Checklist:")
        print("\n".join(checklist))
        print("\nProgress: 2/5 tasks completed\n")

    def _generate_welcome_email(self):
        """Create personalized welcome email"""
        email_template = f"""
        Subject: Welcome to Aniket AI, {self.user_context['name']}!

        Dear {self.user_context['name']},

        We're excited to have you join us as a {self.user_context['role']}!

        Your onboarding schedule:
        - First day orientation: {self.user_context['start_date']}
        - Team meeting: {self.user_context['start_date']} 10:00 AM
        - Equipment setup: IT Department (Floor 3)

        Best regards,
        Aniket AI HR Team
        """
        print(textwrap.dedent(email_template))

    def _log_interaction(self, question: str, answer: str):
        """Maintain interaction history"""
        self.interaction_history.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'user': self.user_context
        })

    # Additional methods for resources/schedule would go here...


if __name__ == "__main__":
    agent = OnboardingAgent()
    agent.start_session()