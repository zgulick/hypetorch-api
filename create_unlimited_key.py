# create_unlimited_key.py
import os
import sys
from dotenv import load_dotenv
from api_key_manager import create_api_key
from token_manager import add_tokens

# Load environment variables
load_dotenv()

# Set client name for the API key
client_name = "HypeTorch Website"
if len(sys.argv) > 1:
    client_name = sys.argv[1]

try:
    # Create the API key with no expiration
    key_info = create_api_key(client_name, None)
    
    # Add a massive number of tokens to this key
    api_key_id = key_info["id"]
    add_tokens(api_key_id, 10000000, "Unlimited tokens for website")
    
    # Get the API key value
    api_key = key_info["api_key"]
    
    print("\n✅ Unlimited API key created successfully!")
    print(f"API Key: {api_key}")
    print(f"ID: {api_key_id}")
    print(f"Client Name: {client_name}")
    
    # Add the key to .env file
    env_file = ".env"
    env_vars = {}
    
    # Read existing variables
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    env_vars[key] = value
    
    # Add or update variables
    env_vars["DEVELOPMENT_MODE"] = "true"
    env_vars["DEV_API_KEY"] = api_key
    
    # Write back to .env file
    with open(env_file, "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    print(f"\n✅ API key saved to {env_file}")
    
    # Create an .env.local file for the frontend
    with open(".env.local", "w") as f:
        f.write(f"NEXT_PUBLIC_API_KEY={api_key}\n")
        f.write("NEXT_PUBLIC_API_URL=https://hypetorch-api.onrender.com/api\n")
    
    print("✅ API key saved to .env.local for frontend use")
    print("\nCopy the .env.local file to your hypetorch-web project directory.")
    
except Exception as e:
    print(f"\n❌ Error creating unlimited API key: {e}")