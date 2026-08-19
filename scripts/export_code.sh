#!/bin/bash
# Export source code for FCIL-AndroidMalware project
# Creates a clean ZIP archive excluding data, checkpoints, and logs

set -e

PROJECT_NAME="fcil_android_malware"
OUTPUT_DIR="./exported_code"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ZIP_NAME="${PROJECT_NAME}_${TIMESTAMP}.zip"

echo "=========================================="
echo "Exporting FCIL-AndroidMalware source code"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create temporary directory for staging
STAGING_DIR=$(mktemp -d)
trap "rm -rf $STAGING_DIR" EXIT

# Copy source files to staging
echo "Copying source files..."

# Create directory structure in staging
mkdir -p "$STAGING_DIR/$PROJECT_NAME"

# Copy all Python files
cp -r config "$STAGING_DIR/$PROJECT_NAME/"
cp -r data "$STAGING_DIR/$PROJECT_NAME/"
cp -r models "$STAGING_DIR/$PROJECT_NAME/"
cp -r incremental "$STAGING_DIR/$PROJECT_NAME/"
cp -r federated "$STAGING_DIR/$PROJECT_NAME/"
cp -r training "$STAGING_DIR/$PROJECT_NAME/"
cp -r utils "$STAGING_DIR/$PROJECT_NAME/"
cp -r experiments "$STAGING_DIR/$PROJECT_NAME/"
cp -r scripts "$STAGING_DIR/$PROJECT_NAME/"

# Copy root files
cp README.md "$STAGING_DIR/$PROJECT_NAME/"
cp requirements.txt "$STAGING_DIR/$PROJECT_NAME/"

# Remove __pycache__ and .pyc files
echo "Cleaning up..."
find "$STAGING_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$STAGING_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$STAGING_DIR" -type f -name "*.pyo" -delete 2>/dev/null || true
find "$STAGING_DIR" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# Create ZIP archive
echo "Creating ZIP archive..."
cd "$STAGING_DIR"
zip -r "$ZIP_NAME" "$PROJECT_NAME" -x "*.git*" "*.DS_Store" "*.egg-info*"

# Move to output directory
mv "$ZIP_NAME" "$OUTPUT_DIR/"

# Create latest symlink
ln -sf "$ZIP_NAME" "$OUTPUT_DIR/${PROJECT_NAME}_latest.zip"

echo ""
echo "=========================================="
echo "Export complete!"
echo "=========================================="
echo "Output: $OUTPUT_DIR/$ZIP_NAME"
echo ""

# List contents
echo "Archive contents:"
unzip -l "$OUTPUT_DIR/$ZIP_NAME" | head -30
echo "..."
echo ""
echo "Total files: $(unzip -l "$OUTPUT_DIR/$ZIP_NAME" | tail -1 | awk '{print $2}')"
echo "Archive size: $(du -h "$OUTPUT_DIR/$ZIP_NAME" | cut -f1)"
