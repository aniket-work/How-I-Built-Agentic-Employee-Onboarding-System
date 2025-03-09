# src/agent.py
"""
Main agent logic for the AI Onboarding System.
"""

import os
from functools import lru_cache
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from src.constants import COLORS, COMMANDS
from src.utils import format_section, load_config, load_settings, get_current_date, get_resources_for_role
from src.vector_db import OnboardingVectorDB

# Load environment variables
load_dotenv()


class OnboardingAgent:
    """Main agent for handling user interactions during the onboarding process."""

    def __init__(self):
        """Initialize the onboarding agent."""
        # Load configuration
        self.config = load_config()
        self.settings = load_settings()

        # Initialize Groq client
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # Initialize vector database
        self.db = OnboardingVectorDB()

        # Initialize user context and interaction history
        self.interaction_history = []
        self.user_context = {}

    def start_session(self):
        """Initialize onboarding session."""
        print(format_section("Welcome to Aniket AI Onboarding System", [], COLORS["title"]))
        self._collect_initial_info()
        self._show_help()
        self._main_interaction_loop()

    def _collect_initial_info(self):
        """Collect user information."""
        print(format_section("Let's Get Started", [], COLORS["success"]))
        self.user_context['name'] = input(f"{COLORS['input']}Your full name: ").strip()
        self.user_context['role'] = input(f"{COLORS['input']}Your job role: ").strip()
        self.user_context['start_date'] = get_current_date()
        print(f"\n{COLORS['success']}Welcome, {self.user_context['name']}! Setting up your onboarding...\n")

    def _main_interaction_loop(self):
        """Handle user commands."""
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
        """Display onboarding progress."""
        # Load checklist from settings
        checklist_template = self.settings['checklists']['default']

        # Format checklist items with colors
        formatted_checklist = []
        completed_count = 0
        total_count = len(checklist_template)

        for item in checklist_template:
            if item['status'] == 'completed':
                formatted_checklist.append(f"{COLORS['success']}✓ {item['task']}")
                completed_count += 1
            else:
                formatted_checklist.append(f"{COLORS['warning']}◻ {item['task']}")

        # Add progress information
        progress = f"{COLORS['header']}Progress: {COLORS['success']}{completed_count}/{total_count} tasks completed"

        print(format_section("Onboarding Checklist", formatted_checklist + [progress]))

    def _show_schedule(self):
        """Display training schedule."""
        # Get schedule from settings
        schedule_template = self.settings['schedules']['default']

        # Format schedule with start date
        schedule = []

        # Day 1
        schedule.append(f"{COLORS['header']}Day 1 ({self.user_context['start_date']}):")
        for activity in schedule_template['day1']:
            schedule.append(f"{COLORS['success']}  {activity['time']} - {activity['activity']}")
        schedule.append("")

        # Day 2
        schedule.append(f"{COLORS['header']}Day 2:")
        for activity in schedule_template['day2']:
            schedule.append(f"{COLORS['border']} {activity['time']} - {activity['activity']}")
        schedule.append("")

        # Day 3
        schedule.append(f"{COLORS['header']}Day 3:")
        for activity in schedule_template['day3']:
            schedule.append(f"{COLORS['warning']} {activity['time']} - {activity['activity']}")

        # Add calendar link
        schedule.append(
            f"{COLORS['border']}Full calendar: {COLORS['text']}https://calendar.{self.settings['company']['domain']}")

        print(format_section("Onboarding Schedule", schedule))

    def _show_resources(self):
        """Show learning resources."""
        role = self.user_context.get('role', 'general').lower()
        resources = get_resources_for_role(role)

        # Format resources for display
        formatted_resources = []
        for i, resource in enumerate(resources, 1):
            formatted_resources.append(f"{COLORS['success']}{i}. {resource['name']}: {COLORS['text']}{resource['url']}")

        print(format_section("Learning Resources", formatted_resources))

    def _generate_welcome_email(self):
        """Generate welcome email."""
        company_name = self.settings['company']['name']

        email_content = [
            f"{COLORS['header']}Subject: Welcome to {company_name}, {self.user_context['name']}!",
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
            f"{COLORS['success']}{company_name} HR Team"
        ]

        print(format_section("Welcome Email", email_content, COLORS["success"]))

    def _show_help(self):
        """Show help menu."""
        help_content = []

        # Format commands
        for cmd, description in COMMANDS.items():
            if cmd in ["exit"]:
                color = COLORS["warning"]
            else:
                color = COLORS["success"] if len(help_content) % 2 == 0 else COLORS["border"]

            help_content.append(f"{color}{cmd.ljust(15)}: {description}")

        print(format_section("Available Commands", help_content))

    @lru_cache(maxsize=100)
    def _handle_question(self, question: str):
        """
        Answer questions using RAG.

        Args:
            question: User's question
        """
        # Query vector database
        rag_config = self.config['rag']
        results = self.db.query_documents(
            query_text=question,
            doc_type=rag_config['default_document_type'],
            n_results=rag_config['query_results']
        )

        # Prepare context for LLM
        context = "\n".join(results['documents'][0]) if results['documents'] else ""

        # Query LLM
        llm_config = self.config['llm']
        response = self.groq.chat.completions.create(
            model=llm_config['model'],
            messages=[{
                "role": "system",
                "content": f"""Answer as HR assistant for {self.user_context['name']}.
                Context: {context}
                Be concise and professional."""
            }, {
                "role": "user",
                "content": question
            }],
            temperature=llm_config['temperature']
        )

        answer = response.choices[0].message.content
        print(format_section("Answer", [answer], COLORS["success"]))