import asyncio
import logging
from copy import deepcopy
from typing import Tuple, List, Dict
from pydantic import BaseModel, Field
from pymilvus import MilvusClient

from utils import embed_text, llm_request
from consts import DB_URI, COLLECTION_NAME


class Conversation(BaseModel):
    history: List[Dict[str, str]] = Field(default=[])
    db_uri: str = Field(default=DB_URI)
    collection_name: str = Field(default=COLLECTION_NAME)


    def retrieve_context(self, query: str) -> str:
        client = MilvusClient(self.db_uri)
        search_res = client.search(
            collection_name=self.collection_name,
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
    

    def get_system_prompt(
            self,
            rag_response: str=""
        ) -> str:
        system = "You are designed to answer questions about Charlie Bell. You may only use information given to you."
        if rag_response:
            system += f"\nYou are given the following factual information on Charlie Bell:\n{rag_response}"
            system += f"\nPlease take into consideration this information if relevant and formulate your response."
        else:
            system += f"\nYou do not have any information about Charlie Bell related to the user's message."

        return system


    def append_message(
            self,
            role: str,
            content: str,
            metadata: dict={}
        ):
        self.history.append({
                "role": role,
                "content": content,
                "metadata": metadata
            })


    def build_conversation(
            self,
            query: str
        ) -> List[Dict[str, str|List[Dict[str, str]]]]:
        conversation = []
        if self.history:
            conversation = [{"role": item["role"], "content": [{"text": item["content"]}]}
                            for item in self.history]
        conversation.append({
            "role": "user",
            "content": [{"text": query}]
        })
        return conversation
        

def main():
    conversation_obj = Conversation()
    while True:
        query = input("Input: ") # """What is Charlie's favorite color?"""

        try:
            # RAG data format
            rag_response = conversation_obj.retrieve_context(query).strip()
            print("--- RAG RESPONSE ---")
            print(rag_response)
            print("--- RAG END ---")
        except Exception as e:
            print("RAG service unexpected exception.")
            logging.exception(e)
            rag_response = ""


        conversation = conversation_obj.build_conversation(query)
        system = conversation_obj.get_system_prompt(rag_response)
        print(conversation, system)
        try:
            tgi_response = llm_request(conversation, system).strip()
            print("--- LLM RESPONSE ---")
            print(tgi_response)
            print("--- LLM END ---")

            # Append User and Assistant messages to conversation history
            conversation_obj.append_message(
                role="user",
                content=query,
                metadata={"system": system}
            )
            conversation_obj.append_message(
                role="assistant",
                content=tgi_response,
                metadata={"system": system}
            )

        except Exception as e:
            print("LLM service unexpected exception.")
            logging.exception(e)
            tgi_response = ""


if __name__ == "__main__":
    main()