#!/bin/bash
# Script to build and upload UnReflectAnything to PyPI
# Usage: ./upload_to_pypi.sh [testpypi|pypi]

set -e  # Exit on error
source .venv/bin/activate
REPO="${1:-testpypi}"  # Default to testpypi for safety

echo "=========================================="
echo "UnReflectAnything PyPI Upload Script"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run this script from the pypi/ directory."
    exit 1
fi

# Install/upgrade build tools
echo "Step 1: Installing/upgrading build tools..."
uv pip install --upgrade build twine --quiet

# Clean previous builds
echo "Step 2: Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

# Build package
echo "Step 3: Building package..."
python -m build

# Check what was built
echo ""
echo "Step 4: Built artifacts:"
ls -lh dist/

# Upload
echo ""
echo "Step 5: Uploading to $REPO..."
if [ "$REPO" = "testpypi" ]; then
    echo "Uploading to TestPyPI (dry run)..."
    python -m twine upload --repository testpypi dist/*
else
    echo "Uploading to PyPI (production)..."
    read -p "Are you sure you want to upload to PyPI? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Upload cancelled."
        exit 0
    fi
    python -m twine upload dist/*
fi

echo ""
echo "=========================================="
echo "Upload complete!"
echo "=========================================="
echo ""
echo "Install with: pip install unreflect-anything"
if [ "$REPO" = "testpypi" ]; then
    echo "Or from TestPyPI: pip install -i https://test.pypi.org/simple/ unreflect-anything"
fi

