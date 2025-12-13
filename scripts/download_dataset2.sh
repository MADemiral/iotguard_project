#!/bin/bash

# CIC-IDS-2017 Dataset Download Script
# Downloads all files from http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/

BASE_URL="http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/"
OUTPUT_DIR="dataset2"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "=================================="
echo "CIC-IDS-2017 Dataset Downloader"
echo "=================================="
echo "Base URL: $BASE_URL"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "This will download all files from CIC-IDS-2017 dataset"
echo ""
echo "Starting download..."
echo ""

# Use wget with recursive download
# -r: recursive download
# -np: don't ascend to parent directory
# -nH: don't create host directory
# --cut-dirs=3: skip first 3 directory levels in the path
# -R "index.html*": reject index.html files
# --progress=dot:giga: show progress in dots (better for large files)
# -e robots=off: ignore robots.txt
# --retry-connrefused: retry even if connection is refused
# --waitretry=1: wait between retries
# --read-timeout=20: timeout for reading data
# --timeout=15: timeout for connections
# -c: continue partial downloads
# -P: specify output directory

wget -r \
     -np \
     -nH \
     --cut-dirs=3 \
     -R "index.html*" \
     --progress=dot:giga \
     -e robots=off \
     --retry-connrefused \
     --waitretry=1 \
     --read-timeout=20 \
     --timeout=15 \
     -c \
     -P "$OUTPUT_DIR" \
     "$BASE_URL"

echo ""
echo "=================================="
echo "Download Complete!"
echo "=================================="
echo ""
echo "Downloaded files are in: $OUTPUT_DIR"
echo ""
echo "Summary of downloaded content:"
find "$OUTPUT_DIR" -type f | wc -l | xargs echo "Total files:"
echo ""
echo "CSV files:"
find "$OUTPUT_DIR" -name "*.csv" | wc -l
echo ""
echo "PCAP files:"
find "$OUTPUT_DIR" -name "*.pcap" -o -name "*.pcapng" | wc -l
echo ""
echo "PDF files:"
find "$OUTPUT_DIR" -name "*.pdf" | wc -l
echo ""
echo "Total size:"
du -sh "$OUTPUT_DIR"
