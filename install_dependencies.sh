#!/bin/bash
set -e

VENV_DIR="venv"

echo "Setting up virtual environment in $VENV_DIR..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

echo "Installing stretch4_emulated_rgbd and dependencies..."
python3 -m pip install -e .

echo "Installation complete!"
echo "To activate the virtual environment for future use, run:"
echo "source $VENV_DIR/bin/activate"
