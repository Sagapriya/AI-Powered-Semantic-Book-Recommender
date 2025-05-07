# Build a Semantic Book Recommender with LLMs – Inspired by FreeCodeCamp

This repo contains all the code for a semantic book recommendation system inspired by the FreeCodeCamp course *"Build a Semantic Book Recommender with LLMs – Full Course"*. While based on the same foundational logic, this version features a **custom interface built with Streamlit**, adding a more interactive user experience.

There are five components to this implementation:

- **Text data cleaning** (code in the notebook `data-exploration.ipynb`)
- **Semantic (vector) search** using LLMs and vector databases (code in the notebook `vector-search.ipynb`). This enables users to find books similar to a natural language query (e.g., “a book about a girl who time travels”).
- **Text classification** using zero-shot classification in LLMs (code in the notebook `text-classification.ipynb`). This allows books to be automatically labeled as genres like "fiction", "non-fiction", or other custom labels.
- **Sentiment analysis** on book descriptions (code in the notebook `sentiment-analysis.ipynb`). This helps classify books by tone (e.g., humorous, suspenseful, joyful).
- **Streamlit web application** for users to search and explore books through a user-friendly interface (`streamlit-dashboard.py`).

---

## Dependencies

This project was built using Python 3.11. To run the code, the following dependencies are required:

- `kagglehub`
- `pandas`
- `matplotlib`
- `seaborn`
- `python-dotenv`
- `langchain`
- `langchain-community`
- `langchain-openai`
- `langchain-chroma`
- `transformers`
- `streamlit`
- `notebook`
- `ipwidgets`

All dependencies are listed in `requirements.txt`.

---

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Configuration:**
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Dataset Access:**
   Download the dataset from [Kaggle](https://www.kaggle.com/) and place the files in the appropriate directory. Instructions are provided in the notebooks.

---

## File Structure

- `data-exploration.ipynb`: Data cleaning and exploration.
- `vector-search.ipynb`: Build and query a vector database.
- `text-classification.ipynb`: Classify book categories using LLMs.
- `sentiment-analysis.ipynb`: Analyze emotional tones in book descriptions.
- `streamlit-dashboard.py`: The main interactive user interface.

---

## Note

This project is **not an official FreeCodeCamp course**. It is inspired by their original tutorial, with significant enhancements including the use of **Streamlit** for the frontend instead of **Gradio**, and customized features for a more interactive recommendation experience.

