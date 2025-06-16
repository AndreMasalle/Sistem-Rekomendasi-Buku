import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import gradio as gr
from config import KEY, TAG, DATASET, NA
import numpy as np

embedding = OpenAIEmbeddings(api_key=KEY)
books = pd.read_csv(DATASET)

books["thumbnail_gede"] = books["thumbnail"] + "&fife=w800"
books["thumbnail_gede"] = np.where(
    books['thumbnail_gede'].isna(),
    NA,
    books['thumbnail_gede']
)

raw_documents = TextLoader(TAG, encoding="utf-8").load()
text_splitter = CharacterTextSplitter(
    separator="\n", chunk_size=0, chunk_overlap=0
)

documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, embedding)


def fetch_rekomendasi(
        query: str,
        category: str = None,
        tone: str = None,
        top_k: int = 50,
        display: int = 16
) -> pd.DataFrame:
    if query:
        rekomendasi = db_books.similarity_search(query, k=top_k)
        list_buku = [int(rek.page_content.strip('"').split()[0]) for rek in rekomendasi]
        rekomendasi_buku = books[books["isbn13"].isin(list_buku)]
    else:
        rekomendasi_buku = books

    if category != "All":
        rekomendasi_buku = rekomendasi_buku[rekomendasi_buku['simple_categories'] == category]
    else:
        rekomendasi_buku = rekomendasi_buku

    if tone == "Happy":
        rekomendasi_buku.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        rekomendasi_buku.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        rekomendasi_buku.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        rekomendasi_buku.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        rekomendasi_buku.sort_values(by="sadness", ascending=False, inplace=True)

    return rekomendasi_buku.head(display)


def rekomen(
        query: str,
        category: str,
        tone: str
):
    rekomendasi = fetch_rekomendasi(query, category, tone)
    results = []

    for _, row in rekomendasi.iterrows():
        deskripsi = row["description"]
        split_desk = deskripsi.split()
        potong_desk = " ".join(split_desk[:30]) + "..."  # jadi kalau deskripsinya lebih dari 30 kata nanti diikuti ...

        authors_raw = row["authors"]
        if isinstance(authors_raw, str):
            author = authors_raw.split(";")
            if len(author) == 2:
                author_str = f"{author[0]} and {author[1]}"
            elif len(author) > 2:
                author_str = f"{', '.join(author[:-1])}, and {author[-1]}"
            else:
                author_str = author[0]
        else:
            author_str = "Unknown Author"


        caption = f"{row['title']} by {author_str}: {potong_desk}"
        results.append((row['thumbnail_gede'], caption))
    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter description of a book:",
                                placeholder="e.g., A book to learn forgiveness ")
        kategori_dropdown = gr.Dropdown(choices=categories,
                                        label="Select a category",
                                        value="All")
        tone_dropdown = gr.Dropdown(choices=tones,
                                    label="Select an emotional tone",
                                    value="All")
        submit_button = gr.Button(value="Find Recommendation")

    gr.Markdown("## Recommendation")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=rekomen,
                        inputs=[user_query, kategori_dropdown, tone_dropdown],
                        outputs=output)
if __name__ == "__main__":
    dashboard.launch()
