import time
import uuid
import streamlit as st
import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
KB_ID = "N7BYBBIO1J"
DATA_SOURCE_ID = "SSNGWXIFVL"
BUCKET_NAME = "rag-uploads-rituraj-1"
MODEL_ARN = "arn:aws:bedrock:us-east-1:958202484330:inference-profile/global.anthropic.claude-haiku-4-5-20251001-v1:0"

# ---------------------------
# Secrets / AWS session
# ---------------------------
required_secrets = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
missing = [k for k in required_secrets if k not in st.secrets]
if missing:
    st.error(f"Missing Streamlit secrets: {', '.join(missing)}")
    st.stop()

aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
aws_session_token = st.secrets.get("AWS_SESSION_TOKEN", None)

session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
    region_name=REGION
)

s3 = session.client("s3")
bedrock_agent = session.client("bedrock-agent")
bedrock_runtime = session.client("bedrock-agent-runtime")

# ---------------------------
# Streamlit page
# ---------------------------
st.set_page_config(page_title="Document Chat on AWS", layout="centered")
st.title("📄 Document Chat on AWS")
st.write("Upload a document, let it sync automatically, and ask questions about that specific document.")

# ---------------------------
# Per-session isolation
# ---------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

SESSION_ID = st.session_state.session_id
PREFIX = f"documents/{SESSION_ID}/"
SOURCE_URI_PREFIX = f"s3://{BUCKET_NAME}/{PREFIX}"

for key, default in {
    "uploaded_file_name": None,
    "s3_key": None,
    "job_id": None,
    "ready": False,
    "sync_status": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.caption(f"Session ID: {SESSION_ID}")

# ---------------------------
# Helpers
# ---------------------------
def normalize_query(q: str) -> str:
    q_clean = q.strip().lower()

    mapping = {
        "summarize": "Provide a detailed summary of the uploaded document.",
        "summary": "Provide a detailed summary of the uploaded document.",
        "key points": "What are the main points of the uploaded document?",
        "main points": "What are the main points of the uploaded document?",
        "about": "What is the uploaded document about?",
        "overview": "Give an overview of the uploaded document.",
        "skills": "List the important skills, competencies, capabilities, or technical terms mentioned in the uploaded document, if any.",
        "experience": "Summarize the important experience, background, or prior work mentioned in the uploaded document, if any.",
        "projects": "List and summarize the important projects, initiatives, studies, or case studies mentioned in the uploaded document, if any.",
        "findings": "What are the main findings or conclusions in the uploaded document?",
        "conclusion": "What conclusion does the uploaded document present?",
        "risks": "What risks, concerns, or challenges are discussed in the uploaded document?",
    }
    return mapping.get(q_clean, q.strip())

def build_fallback_query(original_query: str) -> str:
    filename = st.session_state.get("uploaded_file_name") or "uploaded document"
    return (
        f"You are answering questions about a single uploaded file named '{filename}'. "
        f"Use only that document. "
        f"Read the document carefully and answer this request in a grounded, detailed way: {original_query}"
    )

def upload_file(file):
    key = PREFIX + file.name
    file.seek(0)
    s3.upload_fileobj(
        file,
        BUCKET_NAME,
        key,
        ExtraArgs={"ContentType": file.type or "application/octet-stream"},
    )
    return key

def start_sync():
    response = bedrock_agent.start_ingestion_job(
        knowledgeBaseId=KB_ID,
        dataSourceId=DATA_SOURCE_ID,
        description=f"Streamlit upload sync for session {SESSION_ID}",
    )
    return response["ingestionJob"]["ingestionJobId"]

def get_ingestion_job(job_id):
    response = bedrock_agent.get_ingestion_job(
        knowledgeBaseId=KB_ID,
        dataSourceId=DATA_SOURCE_ID,
        ingestionJobId=job_id,
    )
    return response["ingestionJob"]

def estimate_progress(ingestion_job: dict) -> int:
    status = ingestion_job.get("status", "STARTING")
    stats = ingestion_job.get("statistics", {}) or {}

    scanned = stats.get("numberOfDocumentsScanned", 0) + stats.get("numberOfMetadataDocumentsScanned", 0)
    indexed = stats.get("numberOfNewDocumentsIndexed", 0) + stats.get("numberOfModifiedDocumentsIndexed", 0)
    failed = stats.get("numberOfDocumentsFailed", 0)

    if status == "STARTING":
        return 5
    if status == "IN_PROGRESS":
        if scanned > 0:
            return int(min(90, max(15, ((indexed + failed) / max(scanned, 1)) * 90)))
        return 50
    if status == "COMPLETE":
        return 100
    if status in ("FAILED", "STOPPED", "STOPPING"):
        return 100
    return 0

def retrieval_config():
    return {
        "vectorSearchConfiguration": {
            "numberOfResults": 12,
            "overrideSearchType": "HYBRID",
            "filter": {
                "startsWith": {
                    "key": "x-amz-bedrock-kb-source-uri",
                    "value": SOURCE_URI_PREFIX
                }
            }
        }
    }

def retrieve_only(question: str):
    return bedrock_runtime.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": question},
        retrievalConfiguration=retrieval_config()
    )

def retrieve_and_answer(question: str):
    return bedrock_runtime.retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KB_ID,
                "modelArn": MODEL_ARN,
                "retrievalConfiguration": retrieval_config()
            }
        }
    )

def weak_retrieval(retrieve_response: dict) -> bool:
    results = retrieve_response.get("retrievalResults", [])
    return len(results) == 0

# ---------------------------
# Upload
# ---------------------------
st.header("1) Upload File")
uploaded_file = st.file_uploader("Upload PDF/DOCX/TXT", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    if st.session_state.get("uploaded_file_name") != uploaded_file.name:
        st.session_state.ready = False

    st.success(f"Selected file: {uploaded_file.name}")

    if st.button("Upload & Sync 🚀"):
        try:
            st.session_state.ready = False
            st.session_state.sync_status = None

            with st.spinner("Uploading..."):
                s3_key = upload_file(uploaded_file)
                st.session_state.s3_key = s3_key
                st.session_state.uploaded_file_name = uploaded_file.name

            st.success(f"Uploaded: {s3_key}")

            with st.spinner("Starting sync..."):
                job_id = start_sync()
                st.session_state.job_id = job_id

            progress = st.progress(0, text="Starting sync...")
            status_box = st.empty()
            stats_box = st.empty()

            while True:
                job = get_ingestion_job(job_id)
                status = job.get("status", "STARTING")
                st.session_state.sync_status = status

                pct = estimate_progress(job)
                progress.progress(pct, text=f"Sync status: {status} ({pct}%)")

                stats = job.get("statistics", {}) or {}
                status_box.info(f"Current status: {status}")
                stats_box.caption(
                    f"Scanned: {stats.get('numberOfDocumentsScanned', 0)} docs, "
                    f"{stats.get('numberOfMetadataDocumentsScanned', 0)} metadata | "
                    f"Indexed: {stats.get('numberOfNewDocumentsIndexed', 0)} new, "
                    f"{stats.get('numberOfModifiedDocumentsIndexed', 0)} modified | "
                    f"Failed: {stats.get('numberOfDocumentsFailed', 0)}"
                )

                if status == "COMPLETE":
                    st.session_state.ready = True
                    progress.progress(100, text="Sync complete ✅")
                    st.success("Sync complete. You can ask questions now.")
                    break

                if status in ("FAILED", "STOPPED"):
                    reasons = job.get("failureReasons", [])
                    progress.progress(100, text=f"Sync ended with status: {status}")
                    st.error(f"Sync failed: {status}")
                    if reasons:
                        st.error(" | ".join(reasons))
                    break

                time.sleep(3)

        except ClientError as e:
            st.error(str(e))

if st.session_state.uploaded_file_name:
    st.write(f"Current file: **{st.session_state.uploaded_file_name}**")

if st.session_state.sync_status:
    st.caption(f"Latest sync status: {st.session_state.sync_status}")

# ---------------------------
# Ask
# ---------------------------
st.header("2) Ask Questions")
query = st.text_input("Ask about your document")

if st.button("Ask"):
    if not st.session_state.get("uploaded_file_name"):
        st.warning("Upload file first.")
    elif not st.session_state.get("ready"):
        st.warning("Wait for sync to complete.")
    elif not query.strip():
        st.warning("Enter a question.")
    else:
        try:
            q = normalize_query(query)

            with st.spinner("Checking retrieval..."):
                retrieved = retrieve_only(q)

            if weak_retrieval(retrieved):
                st.warning("⚠️ Weak retrieval detected. Retrying with stronger query...")
                q = build_fallback_query(q)

                with st.spinner("Rechecking retrieval..."):
                    retrieved = retrieve_only(q)

            if weak_retrieval(retrieved):
                st.error("No relevant chunks were retrieved for this question.")
                st.info("Try a more specific prompt like 'what is this document about?' or 'what are the main points?'.")
            else:
                with st.spinner("Generating answer..."):
                    response = retrieve_and_answer(q)

                st.success("Answer")
                st.write(response["output"]["text"])

                citations = response.get("citations", [])
                if citations:
                    st.markdown("### 📚 Sources")
                    shown = 0
                    for citation in citations:
                        for ref in citation.get("retrievedReferences", []):
                            shown += 1
                            text = ref.get("content", {}).get("text", "")
                            location = ref.get("location", {})
                            uri = location.get("s3Location", {}).get("uri", "")

                            with st.expander(f"Source {shown}"):
                                if uri:
                                    st.write(f"**Document:** {uri}")
                                st.write(text if text else "No source text returned.")

                            if shown >= 3:
                                break
                        if shown >= 3:
                            break

        except ClientError as e:
            st.error(str(e))