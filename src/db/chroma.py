import chromadb


class ChromaInstance:
    
    def __init__(self, server_host:str, server_port:int):
        self.host = server_host
        self.port = server_port
        self.client = chromadb.HttpClient(host=self.host, port=self.port)

    def get_embedding_function(self, 
