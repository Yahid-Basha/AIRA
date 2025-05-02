import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Initialize Faiss index
dimension = 384  # Dimension of embeddings from the model
index = faiss.IndexFlatL2(dimension)

# Function to add documents to the Faiss index
def add_to_index(documents):
    embeddings = model.encode(documents)
    index.add(np.array(embeddings, dtype='float32'))

# Streamlit UI
st.title("FAISS Vector Search with Offline Embeddings")

# Input for adding documents
st.subheader("Add Documents")
documents = st.text_area("Enter documents (one per line):")
if st.button("Add to Index"):
    if documents.strip():
        docs_list = documents.split("\n")
        add_to_index(docs_list)
        st.success(f"Added {len(docs_list)} documents to the index.")
    else:
        st.error("Please enter at least one document.")

# Query input for similarity search
st.subheader("Query the Index")
query = st.text_input("Enter your query:")
if st.button("Search"):
    if query.strip():
        query_embedding = model.encode([query])
        distances, indices = index.search(np.array(query_embedding, dtype='float32'), k=5)
        st.write("Top 5 Results:")
        for i, idx in enumerate(indices[0]):
            st.write(f"{i+1}. Document ID: {idx}, Distance: {distances[0][i]}")
    else:
        st.error("Please enter a query.")