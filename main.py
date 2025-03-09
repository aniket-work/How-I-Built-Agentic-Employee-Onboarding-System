# main.py
"""
Entry point for the AI Onboarding System.
"""

import os
import sys
import traceback

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules
from src.agent import OnboardingAgent
from src.utils import ensure_nltk_resources
from src.constants import COLORS


def main():
    """Main entry point for the application."""
    try:
        # Ensure NLTK resources are downloaded
        ensure_nltk_resources()

        # Initialize and start the onboarding agent
        agent = OnboardingAgent()
        agent.start_session()

    except KeyboardInterrupt:
        print(f"\n{COLORS['warning']}Program terminated by user.")
        sys.exit(0)

    except Exception as e:
        print(f"\n{COLORS['warning']}An error occurred: {str(e)}")
        print(f"\n{COLORS['warning']}Error details:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()