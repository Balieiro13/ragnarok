from typing import Protocol

class KBETL(Protocol):
    
    def extract_data(sefl):
        ...
    
    def split_in_chunks(self):
        ...
    
    def embed_data(self):
        ...
    
    def load_data(self):
        ...