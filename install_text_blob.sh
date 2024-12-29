#!/bin/bash
set -euxo pipefail

# Function to log errors but not exit on pre-activation script issues
trap 'echo "Warning: Ignored error in Dataproc pre-activation script";' ERR

# Update package lists
sudo apt-get update

# Install Python3 and pip if not already installed
sudo apt-get install -y python3 python3-pip || echo "Python3 installation failed, continuing..."

# Upgrade pip
pip3 install --upgrade pip || echo "Pip upgrade failed, continuing..."

# Install textblob
pip3 install textblob || echo "TextBlob installation failed, continuing..."

# Reset trap to fail on subsequent errors
trap - ERR