# Projeto Ragnarok

Sugiro criar um virtual environment antes de começar a rodar este projeto:
```
python -m venv .venv
```
Para ativá-lo, dependerá do seu sistema operacional. Ver `https://docs.python.org/3/library/venv.html`

Em seguida, instale as dependências:

```
pip install -r requirements.txt
```

## .env
Edite o `.env` apontando os servidores de text-generation-inference (TGI) e text-embeddings-inference (TEI)

## Instanciando o DB:

Uma vez instalada as bibliotecas e assumindo que vc tenha o Docker instalado (caso não tenha, instale-o), rode de dentro do diretório raíz do projeto:
```
docker-compose up -d
```

## Populando o DB:

Assumindo que vc esteja no diretório raíz deste projeto, para indexar documentos no db, rode o seguinte comando:
```
python geffen/kb_cli.py store /path/to/documents/directory --cn "nome_da_coleção"
```
Note que o caminho precisa terminar no **diretório** no qual seus arquivo em `pdf` estão armazenados. 


## Conversando com os documentos:
Para fazer perguntas aos seus documentos indexados, rode
```
python geffen/geffen_cli.py "Escreva sua pergunta." --cn "nome_da_coleção"
```

## Rodando tudo localmente
Caso você tenha uma GPU da Nvidia com pelo menos 8GB de VRAM, é possível (e recomendável) rodar tudo localmente. Será necessário instalar o [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Uma vez instalados os pré-requisitos, basta alterar o `docker-compose.yml` padrão com o `docker-compose-local.yml`:
```
mv docker-compose-local.yml docker-compose.yml
```
E então, subir os containers:
```
docker-compose up -d
```
Se tudo der certo, finalize alterando o seu `.env` apontando as variáveis para o servidor de embeddings (`http://localhost:8081/`) e o servidor da llm (`http://localhost:8080/`)

