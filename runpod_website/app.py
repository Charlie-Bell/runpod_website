import requests

# Define the target URL
url = "http://172.17.0.2:9200/get-passcode/"

# Define the JSON payload
data = {
    "query": "What is passcode A"
}

# Send the POST request
response = requests.post(url, json=data)

# Check if the request was successful
if response.status_code == 200:
    # Print the response content
    print("Response:", response.json())
else:
    print(f"Failed to get a valid response. Status code: {response.status_code}")
