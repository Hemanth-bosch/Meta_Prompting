"""Simple script to call the Bosch Model Farm (BMF) LLM API."""

import os
from openai import AzureOpenAI

BMF_ENDPOINT = "https://aoai-farm.bosch-temp.com/api"
BMF_DEPLOYMENT = "gpt-5-nano-2025-08-07"
BMF_API_VERSION = "2025-04-01-preview"

api_key = os.environ.get("BMF_API_KEY")
if not api_key:
    raise EnvironmentError("BMF_API_KEY environment variable is not set.")

client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=BMF_ENDPOINT,
    api_version=BMF_API_VERSION,
)

response = client.chat.completions.create(
    model=BMF_DEPLOYMENT,
    messages=[
        {"role": "user", "content": "Hello, who are you?"},
    ],
)

print(response.choices[0].message.content)
