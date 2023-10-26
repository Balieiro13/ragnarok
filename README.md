# RAG App

Por enquanto, a imagem do llama2 quantisado foi gerada localmente através do `openllm` e do `bentoml`. É preciso estudar como configurar um `bentofile.yaml` para gerar a imagem docker do modelo e facilmente rodar este código. Desta forma, estou expondo uma API com o llama2, disponível em `https://abps.ink/llm`

Sugiro criar um virtual environment antes de começar este projeto:
```
$ python -m venv .venv
```
Para ativá-lo, dependerá do seu sistema operacional. Ver `https://docs.python.org/3/library/venv.html`

Em seguida, instale as dependências:

```
$ pip install -r requirements.txt
```

## Instanciando o DB:

Uma vez instalada as bibliotecas e assumindo que vc tenha o Docker instalado (caso não tenha, instale-o), rode de dentro do diretório raíz do projeto:
```
$ docker-compose up -d
```

## Populando o DB:

Assumindo que vc esteja no diretório raíz deste projeto, para indexar documentos no db, rode o seguinte comando:
```
$ python src/store_vectors.py -p /path/to/documents/directory -c "nome_da_coleção"
```

## Conversando com os documentos:
Para fazer perguntas aos seus documentos indexados, rode
```
$ python src/main.py -q "Escreva sua pergunta." -c "nome_da_coleção"
```
Você pode adicionar um `-v` ou `--verbose` para ver todo o prompt gerado pela sua pergunta e seus documentos indexados.
