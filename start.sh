#! /bin/bash

echo ""
echo "Setting up environment"
echo ""

pip install -r requirements.txt
pip install openllm[llms]
