#!/bin/bash

# Setup script for pre-commit hooks
# This script installs the pre-commit framework and sets up hooks for the project

set -e

echo "🔧 Setting up pre-commit hooks for rental-prediction project..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "Makefile" ]; then
    print_error "Makefile not found. Please run this script from the project root."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed."
    exit 1
fi

# Check if pip is available
if ! command -v pip &> /dev/null; then
    print_error "pip is required but not installed."
    exit 1
fi

print_status "Installing pre-commit framework..."
pip install pre-commit

print_status "Installing pre-commit hooks..."
pre-commit install

print_status "Installing commit-msg hook..."
pre-commit install --hook-type commit-msg

print_status "Running pre-commit on all files (first time setup)..."
pre-commit run --all-files

print_success "Pre-commit hooks setup complete! 🎉"
echo ""
echo "Your hooks are now active. They will run automatically on:"
echo "  • git commit (pre-commit hooks)"
echo "  • git commit (commit-msg hook)"
echo ""
echo "To run hooks manually:"
echo "  • pre-commit run --all-files"
echo "  • pre-commit run"
echo ""
echo "To update hooks:"
echo "  • pre-commit autoupdate" 