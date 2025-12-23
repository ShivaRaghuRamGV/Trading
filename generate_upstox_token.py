"""
Upstox Access Token Generator
This script helps you generate an access token for Upstox API
"""

# STEP 1: Fill in your credentials from Upstox Developer Console
API_KEY = "cb681499-48d1-43a4-8a0e-e724d7b9b837"  # Your API key
API_SECRET = "j5l4oap6bs"  # Paste your API secret here
REDIRECT_URI = "http://localhost:8080"  # This should match what you set in Upstox app

print("=" * 70)
print("UPSTOX ACCESS TOKEN GENERATOR")
print("=" * 70)

# Check if credentials are filled
if API_SECRET == "YOUR_API_SECRET_HERE":
    print("\n❌ ERROR: Please fill in your API_SECRET in this file first!")
    print("\nSteps:")
    print("1. Open generate_upstox_token.py")
    print("2. Replace 'YOUR_API_SECRET_HERE' with your actual API secret")
    print("3. Run this script again")
    print("\nGet your API secret from: https://api.upstox.com/")
    input("\nPress Enter to exit...")
    exit(1)

# Generate authorization URL
auth_url = f"https://api.upstox.com/v2/login/authorization/dialog"
params = f"?client_id={API_KEY}&redirect_uri={REDIRECT_URI}&response_type=code"
full_auth_url = auth_url + params

print("\n✅ Step 1: Authorization URL Generated")
print("\n" + "=" * 70)
print("COPY THIS URL AND OPEN IN YOUR BROWSER:")
print("=" * 70)
print(full_auth_url)
print("=" * 70)

print("\n📋 Instructions:")
print("1. Copy the URL above")
print("2. Paste it in your browser and login to Upstox")
print("3. After login, you'll be redirected to a URL like:")
print("   http://localhost:8080/?code=XXXXXX")
print("4. Copy the 'code' parameter from that URL")
print("5. Paste it below")

auth_code = input("\n🔑 Enter the authorization code: ").strip()

if not auth_code:
    print("❌ No code entered. Exiting.")
    exit(1)

# Exchange code for access token
print("\n⏳ Exchanging code for access token...")

import requests

token_url = "https://api.upstox.com/v2/login/authorization/token"
headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/x-www-form-urlencoded'
}
data = {
    'code': auth_code,
    'client_id': API_KEY,
    'client_secret': API_SECRET,
    'redirect_uri': REDIRECT_URI,
    'grant_type': 'authorization_code'
}

try:
    response = requests.post(token_url, headers=headers, data=data)
    
    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data['access_token']
        
        print("\n" + "=" * 70)
        print("✅ SUCCESS! ACCESS TOKEN GENERATED")
        print("=" * 70)
        print("\nYour Access Token:")
        print(access_token)
        print("=" * 70)
        
        # Save to config file
        try:
            with open('upstox_config.txt', 'w') as f:
                f.write(access_token)
            print("\n✅ Token saved to upstox_config.txt")
            print("\nYou can now run the dashboard:")
            print("  python greek_regime_flip_live.py")
        except Exception as e:
            print(f"\n⚠️ Could not save to file: {e}")
            print("\nManually create upstox_config.txt and paste the token above")
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

input("\nPress Enter to exit...")
