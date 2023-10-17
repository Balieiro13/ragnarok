# RAG App

Por enquanto, a imagem do llama2 quantisado que está no `docker-compose.yml` foi gerada localmente atravéz do `openllm` e do `bentoml`. É preciso estudar como configurar um `bentofile.yaml` para gerar a imagem docker do modelo e facilmente rodar este código.

Por enquanto, para conseguir rodá-lo, vc precisará rodar alguns comandos no terminal:

```
$ sh start.sh
```

Uma vez construído o bento, uma tag será gerada. Copie-a e rode
```
$ bentoml containerize thebloke--llama-2-7b-chat-gptq-service:<tag> --opt progress=plain
```

Desta forma, o `bentoml` construirá a imagem docker do llama2 que poderá ser inserida no `docker-compose.yml` da mesma forma: `image: thebloke--llama-2-7b-chat-gptq-service:<tag>`