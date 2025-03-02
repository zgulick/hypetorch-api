import requests
import json

# Path to your JSON data file
json_file_path = "hypetorch_latest_output.json"  # Update this to your file's path

# Load the data
with open(json_file_path, "r") as f:
    data = json.load(f)

# Your API endpoint
upload_url = "https://hypetorch-api.onrender.com/api/upload_json"  # Update with your actual URL

# Upload as a file
files = {"file": (json_file_path, json.dumps(data))}
response = requests.post(upload_url, files=files)

# Print result
print("Response status:", response.status_code)
print("Response content:", response.json())