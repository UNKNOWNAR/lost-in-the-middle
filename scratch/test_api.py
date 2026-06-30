import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(r'e:\WorkSpace\lostinthemiddle\.env'))
keys = [v for k, v in os.environ.items() if k.startswith('GEMINI_API_KEY') and v]

if keys:
    client = genai.Client(api_key=keys[0])
    try:
        response = client.models.generate_content(
            model='gemma-4-31b-it',
            contents='Reply with exactly the word OK'
        )
        print('Response:', response.text)
    except Exception as e:
        print('API Error:', str(e))
else:
    print('No keys found')
