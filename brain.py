# Load environment variables
from dotenv import load_dotenv
import os
import base64
from groq import Groq

load_dotenv()

# Step 1: Setup GROQ API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ Error: GROQ_API_KEY is not set. Please check your .env file.")

print("✅ API Key Loaded Successfully:", GROQ_API_KEY[:5], "... (truncated for security)")

# Step 2: Convert image to required format
image_path = "acne.jpg"

def encode_image(image_path):   
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        print("✅ Image successfully encoded")
        return encoded
    except FileNotFoundError:
        print("❌ Error: Image file not found. Please check the file path:", image_path)
        exit()

encoded_image = encode_image(image_path)

# Step 3: Setup Multimodal LLM 
query = "Can you tell me how to cure acne?"
model = "llama-3.2-90b-vision-preview"

def analyze_image_with_query(query, model, encoded_image):
    print("✅ Setting up Groq API client")

    try:
        client = Groq(api_key=GROQ_API_KEY)  # Ensure API key is passed
    except Exception as e:
        print(f"❌ Error initializing Groq client: {e}")
        exit()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )
        print("✅ API Call Successful")

        response = chat_completion.choices[0].message.content.strip()  # Remove unnecessary spaces
        return response

    except Exception as e:
        print(f"❌ API Error: {e}")
        exit()

# Call the function and print the result
result = analyze_image_with_query(query, model, encoded_image)

if result:
    print("✅ Model Response:\n", result)
else:
    print("❌ No response received from API")
