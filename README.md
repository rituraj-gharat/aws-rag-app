# AWS RAG Document Chat

A multi-user document Q&A system built using:

- AWS Bedrock Knowledge Base
- Streamlit UI
- S3 for document storage

## Features

- Upload your own document
- Auto-sync to Knowledge Base
- Ask questions (RAG)
- Multi-user isolation using session-based prefixing
- Secure AWS integration

## Tech Stack

- Python
- Streamlit
- AWS Bedrock
- S3

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
