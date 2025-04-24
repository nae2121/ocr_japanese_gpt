import os
from openai import OpenAI

client = OpenAI(api_key="Your-OpenAI-API-Key", model="gpt-4o")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "こんにちは！調子はどう？"}
    ]
)

print(response.choices[0].message.content)
