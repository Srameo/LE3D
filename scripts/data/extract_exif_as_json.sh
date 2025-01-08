set -e

DATASET_PATH=$1
SUFFIX=.dng

for file in ${DATASET_PATH}/*${SUFFIX}; do
  echo "Processing $file's exif to json"
  exiftool -json "$file" > "${file%.${SUFFIX}}.json"
done

echo "Done"