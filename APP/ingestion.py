import os
import io
import time
import fitz  # PyMuPDF
from PIL import Image
from google import genai
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

vision_model = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

IMAGE_PROMPT = (
    "Describe this technical diagram, graph, or image in 200 words or less. "
    "Focus on: text, labels, equations, axes, important visual relationships, "
    "the core concept being explained"
)

def process_pdf_multimodal(file_path: str , progress_callback=None) -> list[Document]:
    """
     Extracts text + image descriptions per page.
     progress_callback(current_page, total_pages, message) lets the caller
     (e.g. Streamlit) show live progress instead of only console prints.
    """
    doc = fitz.open(file_path)
    multimodal_pages = []
    image_cache = {}
    total_pages = len(doc)

    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        page_text = page.get_text().strip() or "[No text extracted from page]"

        image_descriptions = []
        image_list = page.get_images(full=True)

        for im_idx, img in enumerate(image_list):
            xref = img[0]

            if xref in image_cache:
                image_descriptions.append(image_cache[xref])
                continue

            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                response = vision_model.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[IMAGE_PROMPT, image]
                )
                description = f"\n[Image {im_idx + 1} Description: {response.text}]\n"
                image_cache[xref] = description
                image_descriptions.append(description)
                if progress_callback:
                    progress_callback(page_num + 1, total_pages,
                                      f"Described image {im_idx + 1} on page {page_num + 1}")
            except Exception as e:
                if progress_callback:
                    progress_callback(page_num + 1, total_pages,
                                      f"Failed image {im_idx + 1} on page {page_num + 1}: {e}")
            finally:
                time.sleep(12)  # rate-limit guard for the vision API

                merged_content = page_text + "\n" + "".join(image_descriptions)
                multimodal_pages.append(Document(
                    page_content=merged_content,
                    metadata={"source": os.path.basename(file_path), "page": page_num + 1}
                ))

                if progress_callback:
                    progress_callback(page_num + 1, total_pages, f"Processed page {page_num + 1}/{total_pages}")

            doc.close()
            return multimodal_pages


def generate_document_summary(merged_documents: list[Document], file_path: str,
                              progress_callback=None) -> Document:
    page_summaries = []
    total = len(merged_documents)

    for i, d in enumerate(merged_documents):
        response = vision_model.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Summarise this page in 150 words or less. Page content:\n{d.page_content}"
        )
        page_summaries.append(response.text)
        time.sleep(12)
        if progress_callback:
            progress_callback(i + 1, total, f"Summarized page {i + 1}/{total}")

    combined_page_summaries = "\n\n".join(page_summaries)

    response = vision_model.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""Create a structured summary.

Include:
1. Main Topic
2. Key Concepts
3. Important Definitions
4. Important Equations
5. Major Conclusions

Page Summaries:
{combined_page_summaries}
"""
    )

    return Document(
        page_content=response.text,
        metadata={"source": os.path.basename(file_path), "type": "summary"}
    )


def ingest_pdf(file_path:str , chunks_store , summary_store, progress_callback=None):
    source_name = os.path.basename(file_path)

    merged_documents = process_pdf_multimodal(file_path, progress_callback)
    summary_doc = generate_document_summary(merged_documents, file_path, progress_callback)

    summary_store.add_documents([summary_doc])

    final_chunks = text_splitter.split_documents(merged_documents)
    chunks_store.add_documents(final_chunks)

    return len(final_chunks), source_name
