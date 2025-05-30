import requests

url = "http://localhost:8000/ask"
payload = {
    "question": " explain how repeated make up cartridges are detected for each Customer ? "
}

response = requests.post(url, json=payload)

# Get response text or JSON string
response_text = response.text  # or use response.json() and convert to string if you want formatted JSON

# Write to response.txt file
with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response_text)

print("Response saved to response.txt")
