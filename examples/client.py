from openai import OpenAI

client = OpenAI(base_url="http://localhost:8788/v1", api_key="your-key")

response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Remember that PostgreSQL 17 is our current database standard."}],
)
print(response.choices[0].message.content)
