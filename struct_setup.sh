#!/bin/bash
# Git Bash commands to create the AI Onboarding project structure

# Create main project directory
mkdir -p ai_onboarding

# Change to project directory
cd ai_onboarding

# Create directory structure
mkdir -p config src src/tests

# Create empty files in the src directory
touch src/__init__.py
touch src/constants.py
touch src/utils.py
touch src/vector_db.py
touch src/agent.py
touch src/tests/__init__.py

# Create config files
touch config/config.json
touch config/settings.yaml

# Create main entry point
touch main.py

# Copy the requirements.txt file
touch requirements.txt

# Initialize git repository
git init

# Create .env file for secrets (not tracked in git)
touch .env
echo "GROQ_API_KEY=your_api_key_here" > .env

# Create .gitignore file
echo "# Python artifacts
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual environment
venv/
ENV/

# Environment variables
.env

# Database
onboarding_db/

# IDE files
.idea/
.vscode/
*.swp
*.swo

# OS specific files
.DS_Store
Thumbs.db" > .gitignore

# Initial git commit
git add .
git commit -m "Initial project structure"

echo "Project structure created successfully in the 'ai_onboarding' directory!"