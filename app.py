"""
AI Document Processor — FastAPI application.

Accepts PDF and image uploads, extracts text via OCR (Tesseract) or
PDF text layers, then uses an LLM to parse the content into structured JSON.
"""

import asyncio
import io
import logging
import os
import uuid
from pathlib import Path

import fitz  # pymupdf
import pytesseract
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from starlette.requests import Request

from processors import DocumentType, extract_fields

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OCR_LANG = os.getenv("OCR_LANG", "eng")  # Tesseract language codes, e.g. "eng+heb+ara"
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "20")) * 1024 * 1024

ALLOWED_MIME_PREFIXES = ("image/", "application/pdf")

app = FastAPI(
    title="AI Document Processor",
    description="Upload documents, extract text via OCR, and get structured data back.",
    version="1.0.0",
)

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


# ---------------------------------------------------------------------------
# OCR / text extraction
# ---------------------------------------------------------------------------

def extract_text_from_image(image_bytes: bytes, lang: str = OCR_LANG) -> str:
    """Run Tesseract OCR on an image and return the extracted text."""
    img = Image.open(io.BytesIO(image_bytes))
    # Convert to RGB if needed (Tesseract handles it better)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return pytesseract.image_to_string(img, lang=lang)


def extract_text_from_pdf(pdf_bytes: bytes, lang: str = OCR_LANG) -> str:
    """
    Extract text from a PDF. Uses the embedded text layer first;
    falls back to OCR on rendered pages when the text layer is empty.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text: list[str] = []

    for page in doc:
        text = page.get_text().strip()
        if text:
            pages_text.append(text)
        else:
            # Render page to image and OCR it
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img, lang=lang)
            pages_text.append(ocr_text)

    doc.close()
    return "\n\n".join(pages_text)


async def run_ocr(file_bytes: bytes, content_type: str, lang: str) -> str:
    """Run the appropriate text extraction in a thread pool (blocking I/O)."""
    loop = asyncio.get_running_loop()
    if content_type == "application/pdf":
        return await loop.run_in_executor(None, extract_text_from_pdf, file_bytes, lang)
    return await loop.run_in_executor(None, extract_text_from_image, file_bytes, lang)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/process")
async def process_document(
    file: UploadFile = File(...),
    lang: str = Query(default=OCR_LANG, description="Tesseract language codes, e.g. eng+heb"),
    doc_type: str | None = Query(default=None, description="Force document type: invoice, contract, receipt, generic"),
):
    """
    Process a single document: extract text via OCR then parse with AI.
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="LLM_API_KEY not configured on server")

    # Validate file type
    ct = file.content_type or ""
    if not any(ct.startswith(prefix) for prefix in ALLOWED_MIME_PREFIXES):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ct}. Send a PDF or image.")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit")

    # Resolve optional forced doc type
    forced_type: DocumentType | None = None
    if doc_type:
        try:
            forced_type = DocumentType(doc_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid doc_type '{doc_type}'. Use: invoice, contract, receipt, generic")

    # OCR
    try:
        extracted_text = await run_ocr(file_bytes, ct, lang)
    except Exception as e:
        logger.exception("OCR failed for %s", file.filename)
        raise HTTPException(status_code=422, detail=f"OCR extraction failed: {e}")

    if not extracted_text.strip():
        return JSONResponse(
            status_code=200,
            content={
                "filename": file.filename,
                "extracted_text": "",
                "result": {"document_type": "unknown", "fields": {}, "error": "No text could be extracted"},
            },
        )

    # AI extraction
    try:
        result = await extract_fields(
            extracted_text,
            doc_type=forced_type,
            api_base=API_BASE,
            api_key=API_KEY,
            model=MODEL,
        )
    except Exception as e:
        logger.exception("AI extraction failed for %s", file.filename)
        raise HTTPException(status_code=502, detail=f"AI extraction failed: {e}")

    return {
        "filename": file.filename,
        "extracted_text": extracted_text[:5000],  # Truncate for response size
        "result": result,
    }


@app.post("/api/batch")
async def batch_process(
    files: list[UploadFile] = File(...),
    lang: str = Query(default=OCR_LANG),
    doc_type: str | None = Query(default=None),
):
    """
    Process multiple documents concurrently. Returns results in the same order.
    Limited to 10 files per request to keep things reasonable.
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch request")

    async def _process_one(f: UploadFile) -> dict:
        """Process a single file, catching errors per-file."""
        try:
            ct = f.content_type or ""
            if not any(ct.startswith(p) for p in ALLOWED_MIME_PREFIXES):
                return {"filename": f.filename, "error": f"Unsupported file type: {ct}"}

            data = await f.read()
            if len(data) > MAX_FILE_SIZE:
                return {"filename": f.filename, "error": "File too large"}

            forced = None
            if doc_type:
                forced = DocumentType(doc_type.lower())

            text = await run_ocr(data, ct, lang)
            if not text.strip():
                return {"filename": f.filename, "extracted_text": "", "result": {"document_type": "unknown", "fields": {}, "error": "No text extracted"}}

            result = await extract_fields(text, doc_type=forced, api_base=API_BASE, api_key=API_KEY, model=MODEL)
            return {"filename": f.filename, "extracted_text": text[:5000], "result": result}
        except Exception as e:
            logger.exception("Batch item failed: %s", f.filename)
            return {"filename": f.filename, "error": str(e)}

    results = await asyncio.gather(*[_process_one(f) for f in files])
    return {"count": len(results), "results": list(results)}


@app.get("/health")
async def health():
    return {"status": "ok", "llm_configured": bool(API_KEY)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
