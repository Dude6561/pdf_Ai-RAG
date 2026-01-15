from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from vector_store import add_document, extract_pdfText, generate_gemini, query_document

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=3)


class Query(BaseModel):
    question: str


# @app.on_event("startup")
# def load_data():
#     add_document(documents)


@app.post("/search")
async def search_docs(q: str, pdf_file: UploadFile = File(...)):
    text = extract_pdfText(pdf_file)
    add_document(text)  # added the document text to vectordb

    result = query_document(q)
    context_text = "\n".join(result)
    prompt = f"""Answer the question using only this context:{context_text} Question: {q}If the context doesn't have the answer, say "I don't know".Answer:"""
    response = await generate_gemini(prompt)
    return {"question": q, "answer": response.text, "context": result}
