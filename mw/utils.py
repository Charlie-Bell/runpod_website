from httpx import AsyncClient


async def post(url, prompt):
    data = {
            "inputs": prompt,
            "parameters": {
                "max_tokens": 250,
                "temperature": 0.95,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 0.8
            }
        }
    headers = {
        "Content-Type": "application/json"
    }

    # Send the POST request
    async with AsyncClient() as client:
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


async def embed_text(url, text):
    data = {
        "inputs": text
    }
    headers = {
        "Content-Type": "application/json"
    }

    async with AsyncClient() as client:
        response = await client.post(url=url, json=data, headers=headers)

    return response.json()[0]


