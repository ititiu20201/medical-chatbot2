import os
import json

def create_directory_structure():
    """Create the project directory structure."""
    directories = [
        'data/raw',
        'data/processed',
        'data/models',
        'src/data',
        'src/models',
        'src/utils',
        'notebooks',
        'configs',
        'tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create __init__.py files in Python package directories
        if directory.startswith('src/'):
            init_file = os.path.join(directory, '__init__.py')
            if not os.path.exists(init_file):
                open(init_file, 'a').close()

def create_config_file():
    """Create initial configuration file."""
    config = {
        "model": {
            "name": "vinai/phobert-base",
            "max_length": 256,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "epochs": 10,
            "warmup_steps": 1000
        },
        "data": {
            "train_file": "data/processed/train.csv",
            "val_file": "data/processed/val.csv",
            "test_file": "data/processed/test.csv"
        },
        "training": {
            "seed": 42,
            "device": "cuda",
            "num_workers": 4,
            "gradient_accumulation_steps": 1
        },
        "paths": {
            "model_save_path": "data/models",
            "log_dir": "logs"
        }
    }
    
    with open('configs/config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def initialize_git():
    """Initialize git repository and create .gitignore."""
    gitignore_content = """
# Python
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
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# VS Code
.vscode/

# PyCharm
.idea/

# Environment
.env
.venv
venv/
ENV/

# Model files
*.pth
*.pt
*.bin

# Logs
logs/
*.log

# Data
data/raw/*
data/processed/*
data/models/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/models/.gitkeep
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    os.system('git init')

def main():
    """Main function to set up the project structure."""
    print("Creating directory structure...")
    create_directory_structure()
    
    print("Creating configuration file...")
    create_config_file()
    
    print("Initializing git repository...")
    initialize_git()
    
    print("Creating placeholder files...")
    # Create placeholder files to preserve directory structure
    placeholders = [
        'data/raw/.gitkeep',
        'data/processed/.gitkeep',
        'data/models/.gitkeep'
    ]
    for placeholder in placeholders:
        open(placeholder, 'a').close()
    
    print("Project structure created successfully!")

if __name__ == "__main__":
    main()