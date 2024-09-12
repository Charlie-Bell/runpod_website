import os
import asyncio
from pymilvus import MilvusClient
from tqdm import tqdm

from utils import embed_text
from consts import DOCS_DIR, DB_URI, COLLECTION_NAME


# Turns docs to a list
def docs_to_list(docs_dir):
    text_lines = []

    for path, subdirs, files in os.walk(docs_dir):
        for name in files:
            relative_path = os.path.join(path, name)
            if relative_path.endswith(".md"):
                with open(relative_path, "r") as f:
                    file_text = f.read()

                text_lines += file_text.split("# ")
    
    return text_lines

# Create the collection
async def create_collection(docs, collection_name=COLLECTION_NAME, uri=DB_URI):
    client = MilvusClient(uri)
    # Delete existing collection
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # Initialize new collection
    embedding_dim = 1024
    client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        metric_type="IP",  # Inner product distance
        consistency_level="Strong",  # Strong consistency level
        auto_id=True,
    )

    # Fill new collection
    data = []
    for i, line in enumerate(tqdm(docs, desc="Creating embeddings")):
        if line:
            data.append({"vector": embed_text(line), "text": line})
    client.insert(collection_name=collection_name, data=data)

    # Close
    client.close()


async def main():
    docs = docs_to_list(DOCS_DIR)
    await create_collection(docs)


if __name__ == "__main__":
    asyncio.run(main())
    

