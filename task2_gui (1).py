import os
import requests
from bs4 import BeautifulSoup
import spacy
import faiss
import numpy as np
from sklearn.preprocessing import normalize
import tkinter as tk
from tkinter import scrolledtext, messagebox


# Step 1: Crawl and scrape website content
def scrape_website(url):
    """Scrape textual content from a website."""
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch {url}: Status code {response.status_code}")
    
    soup = BeautifulSoup(response.content, "html.parser")
    text = " ".join([p.get_text() for p in soup.find_all("p")])  # Extract all text in <p> tags
    return text


# Step 2: Chunk content into smaller segments
def chunk_text(text, chunk_size=300):
    """Split text into chunks of a specified size."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(' '.join(current_chunk)) >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


# Step 3: Generate embeddings using spaCy
def generate_embeddings(chunks, nlp):
    """Generate vector embeddings for each chunk."""
    embeddings = []
    for chunk in chunks:
        doc = nlp(chunk)
        embeddings.append(doc.vector)
    return np.array(embeddings, dtype='float32')


# Step 4: Store embeddings in a FAISS vector database
def store_embeddings(embeddings):
    """Store embeddings in a FAISS index."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    normalized_embeddings = normalize(embeddings)
    index.add(normalized_embeddings)
    return index


# Step 5: Retrieve relevant chunks for a query
def retrieve_relevant_chunks(query, index, nlp, chunks, top_k=3):
    """Retrieve the most relevant chunks based on the query."""
    query_vector = nlp(query).vector.reshape(1, -1)
    query_vector = normalize(query_vector)
    distances, indices = index.search(query_vector, top_k)
    return [chunks[i] for i in indices[0]]


# Main function for RAG pipeline
def rag_pipeline(urls, query):
    """Run the RAG pipeline on given websites and query."""
    # Load spaCy language model
    nlp = spacy.load("en_core_web_md")  # Medium-sized English model with word vectors

    # Scrape, process, and generate embeddings
    all_chunks = []
    for url in urls:
        website_text = scrape_website(url)
        chunks = chunk_text(website_text)
        all_chunks.extend(chunks)

    chunk_embeddings = generate_embeddings(all_chunks, nlp)
    vector_index = store_embeddings(chunk_embeddings)

    # Retrieve relevant chunks for the query
    relevant_chunks = retrieve_relevant_chunks(query, vector_index, nlp, all_chunks)

    # Generate response
    response = "\n".join(relevant_chunks[:3])  # Limit to top 3 relevant chunks
    return response


# GUI Implementation
def run_pipeline():
    """Run the RAG pipeline and display results in the GUI."""
    urls = url_input.get("1.0", tk.END).strip().split("\n")
    query = query_input.get("1.0", tk.END).strip()

    if not urls or not query:
        messagebox.showerror("Error", "Please provide both URLs and a query.")
        return

    try:
        result = rag_pipeline(urls, query)
        result_output.delete("1.0", tk.END)
        result_output.insert(tk.END, result)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


# Create GUI Window
root = tk.Tk()
root.title("RAG Pipeline GUI")
root.geometry("800x600")

# Input Section
tk.Label(root, text="Enter URLs (one per line):", font=("Arial", 12)).pack(pady=5)
url_input = scrolledtext.ScrolledText(root, height=10, width=80, font=("Arial", 10))
url_input.pack(pady=5)

tk.Label(root, text="Enter Query:", font=("Arial", 12)).pack(pady=5)
query_input = scrolledtext.ScrolledText(root, height=5, width=80, font=("Arial", 10))
query_input.pack(pady=5)

# Run Button
run_button = tk.Button(root, text="Run Pipeline", font=("Arial", 12), command=run_pipeline, bg="blue", fg="white")
run_button.pack(pady=10)

# Output Section
tk.Label(root, text="Results:", font=("Arial", 12)).pack(pady=5)
result_output = scrolledtext.ScrolledText(root, height=15, width=80, font=("Arial", 10))
result_output.pack(pady=5)

# Run the GUI
root.mainloop()
