import asyncio
import os
from io import BytesIO

import chromadb
from dotenv import load_dotenv
from fastapi import File, UploadFile
from google import genai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# embedding model loading
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="devops-docs")


load_dotenv()
key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=key)


def add_document(docs):
    embeddings = embedder.encode(docs).tolist()
    ids = [str(i) for i in range(len(docs))]
    collection.add(ids=ids, documents=docs, embeddings=embeddings)


# query the collection chromadb will automatically do it
def query_document(question, k=3):
    q_embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[q_embedding], n_results=k)
    return results["documents"][0]


def extract_pdfText(file: UploadFile = File(...)):
    files_bytes = file.file.read()
    pdf_stream = BytesIO(files_bytes)
    reader = PdfReader(pdf_stream)
    all_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text.append(text)

    return all_text


async def generate_gemini(prompt: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,  # executor
        lambda: client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        ),
    )
