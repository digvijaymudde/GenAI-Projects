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
prompt1 = "hi!"
prompt2 = "Explain time travel in detail"
prompt3 = "Write a 100 words essay on Machine learning"

prompts = [prompt1,prompt2,prompt3]

for prompt in prompts:
    message={
    "role":"user",
    "content":prompt
    }
    messages=[message]
    response = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=50
        )
    usage = response.usage
    print(f"Prompt: {prompt}    Input token:{usage.prompt_tokens}  Completion token: {usage.completion_tokens} Finish reason:{response.choices[0].finish_reason}" )

# Output
#     Prompt: hi!    Input token:37  Completion token: 23 Finish reason:stop  //COMPLETION FINISH 
# Prompt: Explain time travel in detail    Input token:41  Completion token: 50 Finish reason:length //COMPLETION NOT COMPLETED 
# Prompt: Write a 100 words essay on Machine learning    Input token:44  Completion token: 50 Finish reason:length
# print(response.choices[0].message.content)





# sys_message={
#     "role":"system",
#     "content":"You are my girlfriend"
# }
# message={
#     "role":"user",
#     "content":prompt
# }
# messages=[sys_message,message]
# response = client.chat.completions.create(
#     model=model,
#     messages=messages
# )

# print(response.choices[0].message.content)