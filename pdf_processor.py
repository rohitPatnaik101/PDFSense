import fitz
from langchain.docstore.document import Document
import hashlib

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

def process_pdf(pdf_path, rag_pipeline):
    try:
        print(f"Opening PDF: {pdf_path}")  # Debug path
        pdf_document = fitz.open(pdf_path)
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text("text")
            text += page_text + " "
        
        if not text.strip():
            print("No text extracted from PDF!")
            return
        
        content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if rag_pipeline.has_content(content_hash):
            print(f"Skipping {pdf_path} - content already processed.")
            return
        
        chunks = []
        section_counter = 0
        current_section = "Unknown Section"
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            blocks = page.get_text("blocks")

            for block in blocks:
                block_text = block[4].strip()
                if not block_text:
                    continue
                font_size = block[5] if len(block) > 5 else 10
                if font_size > 12 or block_text.isupper():
                    section_counter += 1
                    current_section = block_text[:50]
                    print(f"Section detected: {current_section}")
                    continue
                for i in range(0, len(block_text), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunk = block_text[i:i + CHUNK_SIZE]
                    if chunk:
                        chunks.append(Document(
                            page_content=chunk,
                            metadata={"source": pdf_path, "page": page_num, "section": current_section, "content_hash": content_hash}
                        ))
        print(f"Total chunks created: {len(chunks)}")
        rag_pipeline.add_documents(chunks, content_hash)
        pdf_document.close()
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")