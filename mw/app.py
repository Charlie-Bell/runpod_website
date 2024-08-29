import asyncio
import logging
from pymilvus import MilvusClient

from utils import embed_text, post
from consts import RAG_URL, TGI_URL, DB_URI, COLLECTION_NAME


async def retrieve_context(url, query, collection_name=COLLECTION_NAME, uri=DB_URI):
    client = MilvusClient(uri)
    search_res = client.search(
        collection_name=collection_name,
        data=[
            await embed_text(url, query)
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


def build_prompt(query, rag_response):
    system = "You are designed to answer questions about Charlie Bell. You may only use information given to you."
    if rag_response:
        system += f"\nYou are given the following factual information on Charlie Bell:\n{rag_response}"
        system += f"\nPlease take into consideration this information if relevant and formulate your response."
    else:
        system += f"\nYou do not have any information about Charlie Bell related to the user's message."
    message = f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{query}\n\n"
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 13 Aug 2024

{system}

{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    print("--- PROMPT ---")
    print(prompt)
    print("--- PROMPT END ---")
    return prompt


async def main():
    query = """When did Charlie complete his masters?"""

    try:
        # RAG data format
        rag_response = await retrieve_context(RAG_URL, query)
        print("--- RAG RESPONSE ---")
        print(rag_response)
        print("--- RAG END ---")
    except Exception as e:
        print("RAG service unexpected exception.")
        logging.exception(e)
        rag_response = ""


    prompt = build_prompt(query, rag_response)
    try:
        tgi_response = await post(TGI_URL, prompt)
        tgi_response = tgi_response["generated_text"].strip()
        print("--- TGI RESPONSE ---")
        print(tgi_response)
        print("--- TGI END ---")
    except Exception as e:
        print("TGI service unexpected exception.")
        logging.exception(e)
        tgi_response = ""    


if __name__ == "__main__":
    asyncio.run(main())