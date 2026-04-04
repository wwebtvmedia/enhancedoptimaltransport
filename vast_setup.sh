#!/bin/bash
# vast_setup.sh - Automated dependency installation for Vast.ai containers

echo "🔧 Installing system dependencies..."
apt-get update
apt-get install -y git tmux libgl1-mesa-glx libglib2.0-0 cron

echo "🐍 Installing Python dependencies..."
# If using a virtual environment, activate it here or just install globally in Docker
pip install --upgrade pip
pip install -r requirements.txt

# Make other scripts executable
chmod +x checkpoint_backup.sh
chmod +x vast_start.sh

echo "✅ Setup complete!"
