# 📊 AI Business Insights Engine (RAG-Powered)

An AI-powered application that transforms unstructured customer data into structured business insights using Retrieval-Augmented Generation (RAG).

---

## 🚀 Overview

This project analyzes raw customer complaints and generates:

- 🚨 Key Issues  
- 🔁 Common Patterns  
- 💡 Business Recommendations  

The system uses embeddings and semantic retrieval to ensure outputs are **context-aware and explainable**, rather than purely generative.

---

## 🧠 Key Features

- 📂 Upload customer data (.txt)
- 🤖 AI-powered analysis using LLMs
- 🧾 Structured JSON output (reliable & consistent)
- 🔍 RAG-based retrieval (top-k relevant context)
- ⚠️ Severity classification of issues
- 📊 Visual insights using charts
- 🧠 Explainability via retrieved context

---

## ⚙️ Tech Stack

- **Python**
- **Streamlit** (UI & deployment)
- **OpenAI API** (LLM + embeddings)
- **NumPy** (vector similarity)
- **Matplotlib** (visualizations)

---

## 🧩 Architecture

---

## 🔍 How RAG is Used

Instead of relying solely on LLM responses, this system:

1. Breaks input into chunks  
2. Generates embeddings for each chunk  
3. Retrieves the most relevant chunks based on query similarity  
4. Feeds retrieved context into the LLM  

👉 This ensures:
- Better accuracy  
- Reduced hallucination  
- Explainable outputs  
