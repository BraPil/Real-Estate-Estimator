#!/bin/bash
# Setup script for multi-version demo
# This extracts source code from git tags into demo folders

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Setting up multi-version demo..."
echo "Repository root: $REPO_ROOT"

cd "$REPO_ROOT"

# Create version-specific source directories
echo ""
echo "Extracting V2.5 source code..."
rm -rf "$SCRIPT_DIR/v2.5"
mkdir -p "$SCRIPT_DIR/v2.5"
git archive v2.5 src/ | tar -x -C "$SCRIPT_DIR/v2.5"
echo "  -> Extracted to demo/v2.5/src/"

echo ""
echo "Extracting V3.3.1 source code..."
rm -rf "$SCRIPT_DIR/v3.3"
mkdir -p "$SCRIPT_DIR/v3.3"
git archive v3.3.1 src/ | tar -x -C "$SCRIPT_DIR/v3.3"
echo "  -> Extracted to demo/v3.3/src/"

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. cd demo"
echo "  2. docker-compose -f docker-compose.demo.yml up --build"
echo ""
echo "This will start:"
echo "  - V2.5 API on http://localhost:8001"
echo "  - V3.3 API on http://localhost:8002"
