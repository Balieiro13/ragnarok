#! /bin/bash

echo ""
echo "Setting up environment"
echo ""

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
