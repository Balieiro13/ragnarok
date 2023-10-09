#!/bin/bash

echo ""
echo "Starting ChromaDB"
echo ""

mkdir chromadb
docker pull chromadb/chroma
docker run -d \
    --env-file ./.chroma_env \
    -p 8000:8000 \
    -v ./chromadb:/chroma/chroma chromadb/chroma 