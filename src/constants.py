# src/constants.py
"""
Constants for the AI Onboarding System.
Contains color definitions and other constants used throughout the application.
"""

from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

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

# Command definitions
COMMANDS = {
    "ask": "Get policy answers",
    "checklist": "View progress",
    "resources": "Learning materials",
    "schedule": "Training timeline",
    "email": "Generate welcome email",
    "help": "Show this menu",
    "exit": "End session"
}

# Chunk size for document processing
DEFAULT_CHUNK_SIZE = 3