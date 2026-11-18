#!/usr/bin/env bash

set -e

# 1. Define configuration variables
# Note: Use the API endpoint '/api/storage/' followed by your repository path
ARTIFACTORY_API_URL="https://artifactory.internal:443"
# Note: This is the normal URL used to download the actual files
ARTIFACTORY_DOWNLOAD_URL="https://artifactory.internal:443/artifactory/elint/ai_module_test_files/unknown_emitters/"

TARGET_DIR="data/input_csv_files/pdw_generator/unknown_emitters"
TEMP_JSON="folder_contents.json"

# 2. Create the destination folder if it doesn't exist
if [ ! -d "$TARGET_DIR" ]; then
  echo "📁 Creating missing target folder: $TARGET_DIR"
  mkdir -p "$TARGET_DIR"
fi

echo "🔍 Asking Artifactory for the list of files..."

# 3. Get the list of all files inside the Artifactory folder as JSON text
curl -u "${ARTIFACTORY_USER}:${ARTIFACTORY_TOKEN}" \
     -L "$ARTIFACTORY_API_URL" \
     -o "$TEMP_JSON" --fail

# 4. Read the JSON text to find every file name, then loop through them
# (This uses 'grep' and 'sed' to pull out the filenames easily without needing extra tools)
grep -o '"uri" : "[^"]*"' "$TEMP_JSON" | sed 's/"uri" : "\/\([^"]*\)"/\1/' | while read -r FILE; do

  # Skip folders if any are returned, we only want the files
  if [ -z "$FILE" ] || [[ "$FILE" == *"/"* ]]; then
    continue
  fi

  LOCAL_PATH="$TARGET_DIR/$FILE"
  REMOTE_URL="$ARTIFACTORY_DOWNLOAD_URL/$FILE"

  # 5. Check if the individual file already exists in the GitLab cache
  if [ -f "$LOCAL_PATH" ]; then
    echo "✅ File already exists: $FILE (Skipping)"
  else
    echo "📥 Downloading new file: $FILE..."
    curl -u "${ARTIFACTORY_USER}:${ARTIFACTORY_TOKEN}" \
         -L "$REMOTE_URL" \
         -o "$LOCAL_PATH" --fail
  fi
done

# Clean up the temporary list file
rm "$TEMP_JSON"
echo "✨ All files checked and updated!"
