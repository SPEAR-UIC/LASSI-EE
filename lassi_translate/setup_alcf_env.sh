#!/bin/bash
# Setup script for ALCF Inference Endpoints on Aurora
# Run this ONCE on a LOGIN NODE before submitting jobs

set -e

echo "========================================================================"
echo "ALCF Inference Endpoints - Environment Setup"
echo "========================================================================"
echo "This script sets up the Python environment for ALCF inference."
echo "Run this on a LOGIN NODE (not compute node)."
echo "========================================================================"
echo ""

# Check we're on a login node
if [[ $(hostname) == x* ]]; then
    echo "WARNING: You appear to be on a compute node ($(hostname))"
    echo "This script should be run on a login node (aurora-uan-*)"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Load frameworks module
echo "Loading Aurora frameworks module..."
module load frameworks
echo ""

# Show Python version
echo "Python version: $(python3 --version)"
echo ""

# Set venv path
VENV_PATH="$HOME/alcf_venv"

# Remove old venv if it exists
if [ -d "$VENV_PATH" ]; then
    echo "Removing existing venv at $VENV_PATH..."
    rm -rf "$VENV_PATH"
fi

# Create fresh venv
echo "Creating Python virtual environment at $VENV_PATH..."
python3 -m venv "$VENV_PATH" --system-site-packages

# Activate venv
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install globus_sdk openai

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import globus_sdk; print('✓ globus_sdk version:', globus_sdk.__version__)"
python3 -c "import openai; print('✓ openai version:', openai.__version__)"

# Deactivate venv
deactivate

echo ""
echo "========================================================================"
echo "Setup complete!"
echo "========================================================================"
echo "Virtual environment created at: $VENV_PATH"
echo ""
echo "Next steps:"
echo "1. Download authentication helper:"
echo "   wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py"
echo ""
echo "2. Authenticate with Globus:"
echo "   module load frameworks"
echo "   source $VENV_PATH/bin/activate"
echo "   python3 inference_auth_token.py authenticate"
echo ""
echo "3. Submit test job:"
echo "   qsub lassi_translate/test_alcf_job.pbs"
echo ""
echo "========================================================================"
