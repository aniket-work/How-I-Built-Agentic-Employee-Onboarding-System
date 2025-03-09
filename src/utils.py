# src/utils.py
"""
Utility functions for the AI Onboarding System.
"""

import os
import yaml
import json
import nltk
from typing import List, Dict, Any
from datetime import datetime
from colorama import Style


# Download NLTK data if needed
def ensure_nltk_resources():
    """Ensure required NLTK resources are downloaded."""
    nltk.download('punkt', quiet=True)


def load_config() -> Dict[str, Any]:
    """Load configuration from JSON file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)


def load_settings() -> Dict[str, Any]:
    """Load settings from YAML file."""
    settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'settings.yaml')
    with open(settings_path, 'r') as f:
        return yaml.safe_load(f)


def format_section(title: str, content: list, color) -> str:
    """
    Create consistent left-aligned bordered sections.

    Args:
        title: The title of the section
        content: List of strings to display in the section
        color: Color to use for the border

    Returns:
        Formatted string with border and content
    """
    max_length = max(len(line) for line in content + [title]) if content else len(title)
    border_top = f"{color}╭{'─' * (max_length + 2)}╮"
    title_line = f"{color}│ {Style.BRIGHT}{title.upper().center(max_length)}{Style.NORMAL} {color}│"
    content_lines = []

    from src.constants import COLORS
    for line in content:
        content_lines.append(f"{color}│{COLORS['text']}  {line.ljust(max_length)}  {color}│")

    border_bottom = f"{color}╰{'─' * (max_length + 2)}╯"
    return "\n".join([border_top, title_line] + content_lines + [border_bottom])


def get_current_date() -> str:
    """Get current date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


def get_resources_for_role(role: str) -> List[Dict[str, str]]:
    """
    Get appropriate resources based on user role.

    Args:
        role: User's job role

    Returns:
        List of resource dictionaries
    """
    settings = load_settings()
    resources = settings['resources']['common']

    # Add role-specific resources
    if 'engineer' in role.lower():
        resources.extend(settings['resources']['engineering'])

    return resources