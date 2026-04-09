import pandas as pd
import streamlit as st
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

st.title("📊 AI Customer Insights Dashboard")

# Option selection
option = st.radio("Choose input method:", ["Paste text", "Upload file"])

user_input = ""

# Paste option
if option == "Paste text":
    user_input = st.text_area("Enter customer complaints:")

# Upload option
elif option == "Upload file":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    
    if uploaded_file is not None:
        user_input = uploaded_file.read().decode("utf-8")
        st.text_area("File content:", user_input, height=200)

# Analyze button
if st.button("Analyze") and user_input:

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""
    You are a business analyst.

    Analyze the following customer complaints and return output in JSON:

    {{
      "top_issues": [{{"issue": "", "severity": "High/Medium/Low"}}],
      "patterns": [],
      "recommendations": []
    }}

    Complaints:
    {user_input}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    output = response.choices[0].message.content

    # Clean JSON
    start = output.find("{")
    end = output.rfind("}") + 1
    clean_output = output[start:end]

    data = json.loads(clean_output)

    # Display results
    st.subheader("🔴 Top Issues")
    issues = data["top_issues"]
    # Convert to DataFrame
    df = pd.DataFrame(issues)
    # Display list
    for item in issues:
        st.write(f"- {item['issue']} ({item['severity']})")

    st.subheader("🟡 Patterns")
    for item in data["patterns"]:
        st.write(f"- {item}")

    st.subheader("🟢 Recommendations")
    for item in data["recommendations"]:
        st.write(f"- {item}")

    st.subheader("📊 Issue Severity Distribution")
    # Count severity
    severity_counts = df["severity"].value_counts()
    st.bar_chart(severity_counts)
    st.subheader("📈 Issues Breakdown")
    st.dataframe(df)