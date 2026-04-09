from openai import OpenAI
import os
print("📊 AI Customer Insights Engine\n")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load complaints
with open("data/complaints.txt", "r", encoding="utf-8") as file:
    complaints = file.read()
print("DEBUG:",complaints)
prompt = f"""
You are a business analyst.

Analyze the following customer complaints and return the output STRICTLY in JSON format:

{{
  "top_issues": [],
  "patterns": [],
  "recommendations": []
}}

Complaints:
{complaints}
"""
print(prompt)
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

import json

output = response.choices[0].message.content

start = output.find("{")
end = output.rfind("}") + 1

clean_output = output[start:end]

data = json.loads(clean_output)

print("\n📊 Parsed Data:\n")
print(data)
print("\n📊 AI Customer Insights\n")

print("🔴 Top Issues:")
for issue in data["top_issues"]:
    print(f"- {issue}")

print("\n🟡 Patterns:")
for pattern in data["patterns"]:
    print(f"- {pattern}")

print("\n🟢 Recommendations:")
for rec in data["recommendations"]:
    print(f"- {rec}")
with open("output.txt", "w") as f:
    f.write(str(data))