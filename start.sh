#! /bin/bash

echo ""
echo "Setting up environment"
echo ""

pip install -r requirements.txt
pip install "openllm[gptq]" --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

echo ""
echo "Creating a Bento with OpenLLM "
echo ""

openllm build llama --model-id TheBloke/Llama-2-7B-chat-GPTQ --quantize gptq

echo ""
