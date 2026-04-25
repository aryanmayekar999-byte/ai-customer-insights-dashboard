import streamlit as st
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
from rag import retrieve_similar_documents, retrieve_top_5_similar_documents

# ------------------------
# SETUP
# ------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI Business Insights Engine", layout="wide")

# ------------------------
# HELPER FUNCTIONS
# ------------------------

def chunk_text(text, chunk_size=150):
    sentences = text.split(".")
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += sentence + "."
        else:
            chunks.append(current.strip())
            current = sentence + "."

    if current:
        chunks.append(current.strip())

    return chunks


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_top_k(query_embedding, doc_embeddings, docs, k=3):
    return [
        result["document"]
        for result in retrieve_similar_documents(
            query_embedding,
            doc_embeddings,
            docs,
            k=k,
        )[:k]
    ]


def assign_severity(issue):
    issue = issue.lower()
    if "crash" in issue or "failure" in issue:
        return "High"
    elif "slow" in issue or "delay" in issue:
        return "Medium"
    else:
        return "Low"


def plot_issues(issues):
    counts = [len(issue.split()) for issue in issues]

    plt.figure()
    plt.barh(issues, counts)
    plt.xlabel("Impact Score")
    plt.title("Issue Importance")
    st.pyplot(plt)


def analyze_text(text):
    prompt = f"""
You are a business analyst.

Return STRICT JSON in this format:

{{
  "top_issues": ["...", "...", "..."],
  "patterns": ["...", "..."],
  "recommendations": ["...", "..."]
}}

Data:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# ------------------------
# UI
# ------------------------

st.title("📊 AI Business Insights Engine (RAG-Powered)")
st.markdown("Upload customer complaints and extract actionable insights using AI.")

tab1, tab2, tab3 = st.tabs(["📂 Upload", "📊 Insights", "🔍 RAG Context"])

# ------------------------
# TAB 1: Upload
# ------------------------

with tab1:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        st.session_state["text"] = text
        st.success("File uploaded successfully!")

        st.subheader("Preview")
        st.write(text[:500] + "...")

# ------------------------
# TAB 2: Insights
# ------------------------

with tab2:

    text = st.session_state.get("text")

    if text:

        st.subheader("🤖 AI Analysis")

        # Step 1: Generate result
        result = analyze_text(text)

        # Optional debug
        # st.write(result)

        # Step 2: Parse result
        from utils import safe_parse_json
        parsed = safe_parse_json(result)

        # Step 3: Handle parsing failure
        if parsed is None:
            st.error("⚠️ Could not parse structured output")
            st.write(result)
        elif not parsed["top_issues"]:
            st.warning("⚠️ No issues detected in input data")
        else:
            # Step 4: Display metrics
            col1, col2, col3 = st.columns(3)

            col1.metric("Issues", len(parsed["top_issues"]))
            col2.metric("Patterns", len(parsed["patterns"]))
            col3.metric("Recommendations", len(parsed["recommendations"]))

            # Step 5: Display sections
            st.subheader("🚨 Top Issues")
            for issue in parsed["top_issues"]:
                st.error(issue)

            st.subheader("🔁 Patterns")
            for pattern in parsed["patterns"]:
                st.warning(pattern)

            st.subheader("💡 Recommendations")
            for rec in parsed["recommendations"]:
                st.success(rec)

# ------------------------
# TAB 3: RAG CONTEXT
# ------------------------

with tab3:
    if "text" in st.session_state:

        text = st.session_state["text"]

        st.subheader("🔍 Retrieval Process")

        chunks = chunk_text(text)

        with st.spinner("Generating embeddings..."):
            doc_embeddings = [get_embedding(chunk) for chunk in chunks]

        query = st.text_input("Ask a question about the data:")

        if query:
            query_embedding = get_embedding(query)
            try:
                top_chunks = retrieve_top_5_similar_documents(
                    query_embedding,
                    doc_embeddings,
                    chunks,
                )
            except ImportError as error:
                st.error(str(error))
                top_chunks = []

            st.subheader("📌 Most Relevant Context")

            for result in top_chunks:
                st.info(f"Score: {result['score']:.4f}\n\n{result['document']}")
