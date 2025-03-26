# add_dev_key.py
import os
from dotenv import load_dotenv
from api_key_manager import create_api_key
from token_manager import add_tokens

# Load environment variables
load_dotenv()

# Create a development API key
try:
    # Create the API key
    key_info = create_api_key("Development", None)  # No expiration
    
    # Add a large number of tokens to this key
    api_key_id = key_info["id"]
    add_tokens(api_key_id, 1000000, "Initial development tokens")
    
    print("✅ Development API key created successfully:")
    print(f"API Key: {key_info['api_key']} (SAVE THIS - IT WON'T BE SHOWN AGAIN)")
    print(f"ID: {key_info['id']}")
    
    # Save to .env.local file for frontend
    with open(".env.local", "w") as f:
        f.write(f"NEXT_PUBLIC_API_KEY={key_info['api_key']}\n")
        f.write(f"NEXT_PUBLIC_API_URL=https://hypetorch-api.onrender.com/api\n")
    
    print("✅ API key saved to .env.local file for frontend use")
    
except Exception as e:
    print(f"❌ Error creating development API key: {e}")