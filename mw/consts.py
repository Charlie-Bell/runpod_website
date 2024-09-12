from dotenv import load_dotenv
from os import getenv

load_dotenv()


AWS_ACCESS_KEY = getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = getenv("AWS_SECRET_ACCESS_KEY")

DOCS_DIR = getenv("DOCS_DIR")
DB_URI = getenv("DB_URI")
COLLECTION_NAME = getenv("COLLECTION_NAME")