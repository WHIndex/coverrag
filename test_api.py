import os
import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")

resp = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "请用一句话回答：你能正常工作吗？"}
    ],
    temperature=0,
    max_tokens=50,
)

print(resp["choices"][0]["message"]["content"])
