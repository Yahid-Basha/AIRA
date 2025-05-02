import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ✅ 1. Cache the client — only initialized once per session
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(":memory:")

client = get_qdrant_client()

# ✅ 2. Function to upload only once per session
def upload_points_once(documents, encoder, collection_name="my_collection"):
    if not documents:
        st.error("❌ No documents provided. Please provide valid documents to upload.")
        return

    if "points_uploaded" not in st.session_state:
        print("Initializing session state for points_uploaded.")
        st.session_state.points_uploaded = False

    if not st.session_state.points_uploaded:
        try:
            points = []
            for idx, doc in enumerate(documents):
                if "text" not in doc or not doc["text"]:
                    st.warning(f"⚠️ Skipping document {idx} due to missing or empty 'text'.")
                    continue

                vector = encoder.encode(doc["text"])
                points.append(models.PointStruct(id=idx, vector=vector, payload=doc))

            if not points:
                st.error("❌ No valid documents to upload after preprocessing.")
                return

            vector_size = len(points[0].vector)  # Explicitly set vector size

            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )

            client.upload_points(collection_name=collection_name, points=points)
            st.session_state.points_uploaded = True
            st.success("✅ Points uploaded to in-memory Qdrant.")
        except Exception as e:
            st.error(f"❌ Failed to upload points: {e}")
    else:
        print("Points already uploaded in this session.")
        st.info("ℹ️ Points already uploaded in this session.")

# ✅ 3. Example usage
documents = [{"text": "Streamlit is awesome!"}, {"text": "Qdrant rocks!"}]

# Dummy encoder just for testing
@st.cache_resource
class DummyEncoder:
    def encode(self, text):
        return [float(len(text))] * 5  # 5-dim vector based on text length

encoder = DummyEncoder()

upload_points_once(documents, encoder)
