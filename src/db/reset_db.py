import os
from dotenv import load_dotenv

from db import ChromaControl

load_dotenv()

def main():
    db = ChromaControl(
        server_host = os.getenv("DB_HOST"),
        server_port = os.getenv("DB_PORT"),
        config={"allow_reset": True}
    )

    print("Reseting Database...")
    db.reset_chroma()
    
if __name__=="__main__":
    main()
