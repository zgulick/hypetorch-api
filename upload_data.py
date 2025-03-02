import requests
import json

# Load your JSON data
with open("hypetorch_latest_output.json", "r") as f:
    data = json.load(f)

# Convert to file format for upload
files = {"file": ("data.json", json.dumps(data))}

# Upload to your API
response = requests.post("https://hypetorch-api.onrender.com/api/upload_json", files=files)
print(response.json())