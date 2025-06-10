#!/bin/bash

set -e  # Exit on error

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install google-adk if not present
if ! pip show google-adk > /dev/null 2>&1; then
  echo "Installing google-adk..."
  pip install google-adk
fi

# Run the app
adk web
