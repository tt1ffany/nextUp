import pandas as pd
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800" # Get largest size for best resolution
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "no_cover.jpg", # If thumbnail image not available, use interim cover
    books["large_thumbnail"], # Otherwise use cover from link
)

# Build vector database
raw_documents = TextLoader("tagged_description.txt").load() # Read tagged descriptions into text loader
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0) # Instantiate character text splitter separated by new line
documents = text_splitter.split_documents(raw_documents) # Apply to each document to get document chunks (individual book descriptions)
db_books = Chroma.from_documents(documents, HuggingFaceEmbeddings()) # Convert chunks into document embeddings and store in Chroma vector db

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50, # Initial 50 recs to filter
        final_top_k: int = 16, # 16 fits well in display
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs] # Get ISBNs of recommended books
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k) # Limit dataframe to just match ISBNs of recommended books

    # Dropdown filter
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category][:final_top_k]
    else:
        book_recs = book_recs.head(final_top_k)

    # Sort recommendations based on probability of that emotion
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    if tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    if tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    if tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    if tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs