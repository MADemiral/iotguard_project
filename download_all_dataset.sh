#!/bin/bash

# CIC IoT Dataset 2023 Complete Download Script
# Downloads all files from http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/Dataset/

BASE_URL="http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/Dataset/"
OUTPUT_DIR="dataset"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "=================================="
echo "CIC IoT Dataset 2023 Downloader"
echo "=================================="
echo "Base URL: $BASE_URL"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "This will download:"
echo "  - PRIORITY: All CSV files first"
echo "  - Then: PCAP files, README and supplementary materials"
echo "  - Will skip files that already exist"
echo ""
echo "Starting CSV download first..."
echo ""

# PHASE 1: Download CSV files first
# -r: recursive download
# -np: don't ascend to parent directory
# -nH: don't create host directory
# --cut-dirs=3: skip first 3 directory levels in the path
# -R "index.html*": reject index.html files
# -A "*.csv": accept ONLY CSV files
# --progress=dot:giga: show progress in dots (better for large files)
# -e robots=off: ignore robots.txt
# --retry-connrefused: retry even if connection is refused
# --waitretry=1: wait between retries
# --read-timeout=20: timeout for reading data
# --timeout=15: timeout for connections
# -c: continue partial downloads (skip existing files)
# -nc: no clobber - don't download if file exists
# -P: specify output directory

echo "PHASE 1: Downloading CSV files..."
wget -r \
     -np \
     -nH \
     --cut-dirs=3 \
     -R "index.html*" \
     -A "*.csv" \
     --progress=dot:giga \
     -e robots=off \
     --retry-connrefused \
     --waitretry=1 \
     --read-timeout=20 \
     --timeout=15 \
     -c \
     -nc \
     -P "$OUTPUT_DIR" \
     "${BASE_URL}CSV/"

echo ""
echo "CSV download complete!"
echo ""
echo "PHASE 2: Downloading other files (PCAP, PDF, etc.)..."
echo ""

# PHASE 2: Download everything else
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
     -nc \
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
