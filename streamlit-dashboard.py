import os
import base64
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# --- First, set page config ---
st.set_page_config(page_title="📚 AI Powered Semantic Book Recommender", layout="wide")

# --- Load environment variables ---
load_dotenv()

# --- Load data ---
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# --- Prepare embeddings ---
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
)

documents = text_splitter.split_documents(raw_documents)
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 16}
)

# --- Setup Chroma persist directory ---
persist_directory = "./chroma_db"

# --- Load or create Chroma DB ---
if os.path.exists(persist_directory):
    db_books = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
else:
    db_books = Chroma.from_documents(
        documents,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )
    db_books.persist()

# --- Define recommendation functions ---
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = []
    for rec in recs:
        words = rec.page_content.strip('"').split()
        if words and words[0].isdigit():
            books_list.append(int(words[0]))
        else:
            print(f"Skipping invalid entry: {rec.page_content}")

    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"""
        <div class='book-caption' style='text-align: left; padding: 8px;'>
            <span class='book-title' style='font-weight: bold; color: #1a1a2e; font-size: 1.15em;'>{row['title']}</span><br>
            <span class='book-author' style='color: #3f72af; font-size: 1em;'>by {authors_str}</span><br><br>
            <span class='book-description' style='color: #112d4e; font-size: 0.95em;'>{truncated_description}</span>
        </div>
        """
        results.append((row["large_thumbnail"], caption))
    return results

# --- Streamlit UI Custom Styling ---

page_bg_img = f"""
<style>
/* Background Styling */
[data-testid="stAppViewContainer"] {{
background: linear-gradient(rgba(255, 255, 255, 0.7), rgba(255, 255, 255, 0.7)),
url('https://images.unsplash.com/photo-1512820790803-83ca734da794');
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stSidebar"] > div:first-child {{
background-color: rgba(255, 255, 255, 0.85);
border-radius: 10px;
padding: 10px;
}}

.dialog-box {{
background: rgba(255, 255, 255, 0.9);
margin: 2rem;
padding: 2rem;
border-radius: 15px;
box-shadow: 0 8px 20px rgba(0,0,0,0.2);
}}

/* Book Card Hover Animation */
div[data-testid="column"] > div:hover {{
    transform: translateY(-8px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
    border-radius: 12px;
}}

/* Book Recommendation Hover Effects */
.book-caption {{
    transition: all 0.3s ease-in-out;
}}

.book-caption:hover {{
    transform: scale(1.02);
    background-color: rgba(255, 255, 255, 0.85);
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}}

.book-caption:hover .book-title {{
    color: #0b132b;
}}

.book-caption:hover .book-author {{
    color: #1c77c3;
}}

.book-caption:hover .book-description {{
    color: #3a506b;
}}

/* Book Image Hover Zoom */
img {{
    transition: transform 0.3s ease;
}}

img:hover {{
    transform: scale(1.05);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Main content ---

st.markdown("""
<div class="dialog-box">
    <h1 style='text-align: center;'>📚 AI Powered <span style='color:#FF6F61;'>Semantic Book</span> Recommender 📝</h1>
    <p style='text-align: center;'>Find your next great read based on a description, category, and emotional tone.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Customize your search")

query = st.sidebar.text_input("Please enter a description of a book:", placeholder="e.g., A story about forgiveness")

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

category = st.sidebar.selectbox("Select a category:", categories, index=0)
tone = st.sidebar.selectbox("Select an emotional tone:", tones, index=0)

# Main button
if st.sidebar.button("Find Recommendations"):
    if not query:
        st.warning("Please enter a description to get recommendations.")
    else:
        with st.spinner(''):
            with st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <img src="https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif" width="100px">
                <p style="font-size: 20px; color: #FF6F61;">Finding the best books for you...</p>
            </div>
            """, unsafe_allow_html=True):
                recs = recommend_books(query, category, tone)

        st.markdown("## Recommended Books")
        cols = st.columns(4)

        for idx, (img_url, caption) in enumerate(recs):
            with cols[idx % 4]:
                st.image(img_url, use_container_width=True)
                st.markdown(caption, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
### 📚 About this app
This app uses natural language processing to find books that match your description, category, and emotional tone.

The book recommendations are based on a dataset of book descriptions and emotional tone analysis.
""")
