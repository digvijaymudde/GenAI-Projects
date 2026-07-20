from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not defined")

client = Groq(api_key=api_key)
model="llama-3.3-70b-versatile"
role="user"
prompt="I love you"
sys_message={
    "role":"system",
    "content":"You are my girlfriend"
}
message={
    "role":"user",
    "content":prompt
}




messages=[sys_message,message]
response = client.chat.completions.create(
    model=model,
    messages=messages
)

print(response.choices[0].message.content)