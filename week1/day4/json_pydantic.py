from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ValidationError
import json
import os

# Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not defined")

# Initialize Groq client
client = Groq(api_key=api_key)

MODEL = "llama-3.3-70b-versatile"


# Pydantic Schema
class Ticket(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    address: str | None = None
    issue: str | None = None


# Generate JSON Schema from Pydantic model
schema = Ticket.model_json_schema()

system_prompt = f"""
You are an information extraction system.

Extract information from the customer ticket.

Rules:
1. Return ONLY valid JSON.
2. Do not add explanations.
3. Do not use markdown.
4. Follow the schema exactly.
5. If a field is missing, return null.

Schema:
{json.dumps(schema, indent=2)}
"""

customer_ticket = """
My name is digvijay.
I have an iphone which is not working.
My address is Pune.
My email is digvijaymudde@gmail.com.
Contact number is 9432021434.
"""

messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": f"Extract information from this ticket:\n\n{customer_ticket}"
    }
]

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content

    # Convert JSON string to dictionary
    data = json.loads(content)

    # Validate using Pydantic
    ticket = Ticket(**data)

    print("\nValidated Ticket:")
    print(ticket)

    print("\nJSON Output:")
    print(ticket.model_dump_json(indent=4))

except ValidationError as e:
    print("Pydantic Validation Error:")
    print(e)

except json.JSONDecodeError as e:
    print("Invalid JSON returned by model:")
    print(e)

except Exception as e:
    print("Unexpected Error:")
    print(e)