import httpx
import asyncio


# Define the target URLs
RAG_URL = "https://66qywxhnud1i70-9200.proxy.runpod.net/get-passcode/"
TGI_URL = "https://kdk24nrq7ugtn3-8000.proxy.runpod.net/generate"


async def post(url, data):

    headers = {
        "Content-Type": "application/json"
    }

    # Send the POST request
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json=data,
            headers=headers,
            timeout=10
        )

    # Check if the request was successful
    if response.status_code == 200:
        # Print the response content
        print("Response:", response.json())
    else:
        print(f"Failed to get a valid response. Status code: {response.status_code}")

    return response.json()


def build_prompt(query, rag_response):
    EOT_TOKEN = "<|end_of_turn|>"
    USER_TOKEN = "GPT4 Correct User"
    BOT_TOKEN = "GPT4 Correct Assistant"

    system = "You are a helpful assistant."
    if rag_response:
        system += f"\nYou are given the following context:\n{rag_response}"
    system = system + EOT_TOKEN
    prompt = system + f"{USER_TOKEN}: {query}{EOT_TOKEN}{BOT_TOKEN}:"

    return prompt


async def main():
    query = "What is password A?"

    try:
        # RAG data format
        data = {
            "query": query
        }
        rag_response = await post(RAG_URL, data)
        rag_response = rag_response["context"]
        print("RAG response:", rag_response)
    except Exception as e:
        print("RAG service unexpected exception.")
        print(e)
        rag_response = ""

    prompt = build_prompt(query, rag_response)
    print(prompt)
    try:
        data = {
            "inputs": prompt,
            "parameters": {
                "max_tokens": 50,
                "temperature": 0.95,
                "top_p": 0.9,
                "tok_k": 30,
                "repetition_penalty": 1.1
            }
        }
        tgi_response = await post(TGI_URL, data)
        tgi_response = tgi_response["generated_text"].strip()
        print("TGI Response:", tgi_response)
    except Exception as e:
        print("TGI service unexpected exception.")
        print(e)
        tgi_response = ""    


if __name__ == "__main__":
    asyncio.run(main())