import json
from botocore.exceptions import ClientError

from clients import boto_client


def llm_request(conversation, system, client=boto_client, instruct_model_id="meta.llama3-1-8b-instruct-v1:0") -> str:
    try:
        response = client.converse(
            modelId=instruct_model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": 512, "temperature": 0.9, "topP": 0.9},
            system=[{"text": system}]
        )

        response_text = response["output"]["message"]["content"][0]["text"]
        return response_text

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{instruct_model_id}'. Reason: {e}")
        exit(1)


def embed_text(text, client=boto_client, embedding_model_id="amazon.titan-embed-text-v2:0"):
    body = json.dumps({
        "inputText": text
    })
    accept = "application/json"
    content_type = "application/json"

    response = client.invoke_model(
        body=body,
        modelId=embedding_model_id,
        accept=accept,
        contentType=content_type
    )

    response_body = json.loads(response['body'].read())
    embedding = response_body.get("embedding")
    return embedding