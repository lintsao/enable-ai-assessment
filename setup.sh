#!/bin/bash

set -e  # Exit immediately on error

VENV_DIR="mouth_open_detector"

# Step 1: Create virtual environment if not exists
if [ ! -d "$VENV_DIR" ]; then
  echo "🔧 Creating virtual environment: $VENV_DIR"
  python3.11 -m venv "$VENV_DIR"
else
  echo "✅ Virtual environment already exists: $VENV_DIR"
fi

# Step 2: Activate the virtual environment
echo "📦 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Step 3: Install dependencies
echo "📦 Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🎉 Setup completed. You are now ready to use the virtual environment '$VENV_DIR'"