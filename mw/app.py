import asyncio
import logging
from pymilvus import MilvusClient

from utils import embed_text, llm_request
from consts import DB_URI, COLLECTION_NAME


def retrieve_context(query, collection_name=COLLECTION_NAME, uri=DB_URI):
    client = MilvusClient(uri)
    search_res = client.search(
        collection_name=collection_name,
        data=[
            embed_text(query)
        ],  # Use the `embed_text` function to convert the question to an embedding vector
        limit=3,  # Return top 3 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text"],  # Return the text field
    )

    # Get a list of (text, distance)
    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]

    # Join the 3 text results into a context
    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )

    return context


def get_prompt(query, role, rag_response=""):
    system = "You are designed to answer questions about Charlie Bell. You may only use information given to you."
    if rag_response:
        system += f"\nYou are given the following factual information on Charlie Bell:\n{rag_response}"
        system += f"\nPlease take into consideration this information if relevant and formulate your response."
    else:
        system += f"\nYou do not have any information about Charlie Bell related to the user's message."
    
    conversation = [
        {
            "role": role,
            "content": [{"text": query}]
        }
    ]

    return conversation, system


def main():
    query = """When did Charlie complete his masters?"""

    try:
        # RAG data format
        rag_response = retrieve_context(query)
        print("--- RAG RESPONSE ---")
        print(rag_response)
        print("--- RAG END ---")
    except Exception as e:
        print("RAG service unexpected exception.")
        logging.exception(e)
        rag_response = ""


    conversation, system = get_prompt(query, role="user", rag_response=rag_response)
    try:
        tgi_response = llm_request(conversation, system)
        print("--- LLM RESPONSE ---")
        print(tgi_response)
        print("--- LLM END ---")
    except Exception as e:
        print("LLM service unexpected exception.")
        logging.exception(e)
        tgi_response = ""    


if __name__ == "__main__":
    main()