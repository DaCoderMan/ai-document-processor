# AI Document Processor

A FastAPI service that extracts structured data from documents using OCR and LLM-based parsing. Upload a PDF or image, get clean JSON back with the relevant fields extracted.

## What It Does

1. **Accepts** PDF, PNG, JPG, TIFF uploads via REST API or web UI
2. **Extracts text** using Tesseract OCR (with multi-language support) or PDF text layers
3. **Classifies** the document type automatically (invoice, receipt, contract, or generic)
4. **Parses** the text into structured JSON using any OpenAI-compatible LLM API
5. **Returns** clean, typed fields — amounts as numbers, dates in ISO format, null for missing data

### Supported Document Types

| Type | Extracted Fields |
|------|-----------------|
| **Invoice** | Invoice number, dates, vendor/bill-to info, currency, line items with quantities and prices, subtotal/tax/total |
| **Receipt** | Store name/address, date, itemized list, subtotal, tax, total, payment method |
| **Contract** | Title, parties and roles, effective/expiration dates, key terms, governing law, signatures |
| **Generic** | Title, date, summary, key entities, key data points |

## Setup

### Prerequisites

- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and on PATH
- An API key for any OpenAI-compatible LLM service

### Install

```bash
cd ai-document-processor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file:

```env
LLM_API_KEY=sk-your-api-key
LLM_API_BASE=https://api.openai.com/v1    # or any compatible endpoint
LLM_MODEL=gpt-4o-mini                      # or claude-3-haiku, etc.
OCR_LANG=eng                               # tesseract language codes
MAX_FILE_SIZE_MB=20
```

### Run

```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Open `http://localhost:8080` for the web UI.

## API Reference

### `POST /api/process`

Process a single document.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `file` | form-data | The document file (required) |
| `lang` | query | Tesseract language codes, e.g. `eng+heb` (default: `eng`) |
| `doc_type` | query | Force document type: `invoice`, `receipt`, `contract`, `generic` (default: auto-detect) |

**Response:**
```json
{
  "filename": "invoice-042.pdf",
  "extracted_text": "INVOICE #042...",
  "result": {
    "document_type": "invoice",
    "fields": {
      "invoice_number": "042",
      "date": "2026-03-01",
      "vendor": { "name": "Acme Corp", "address": "123 Main St" },
      "total": 1500.00,
      "line_items": [
        { "description": "Consulting", "quantity": 10, "unit_price": 150.00, "amount": 1500.00 }
      ]
    }
  }
}
```

### `POST /api/batch`

Process up to 10 documents concurrently. Same query parameters as `/api/process`, but send multiple files under the `files` field.

**Response:**
```json
{
  "count": 3,
  "results": [ ... ]
}
```

### `GET /health`

Returns service status and whether the LLM key is configured.

## Web UI

The built-in UI at `/` supports:

- Drag-and-drop or click-to-browse file selection
- Multiple file upload
- Language and document type selection
- Collapsible result cards showing extracted fields and raw OCR text

## Architecture

```
Upload → File validation → OCR (Tesseract / PDF text layer)
       → LLM classification → LLM field extraction → JSON response
```

- **OCR** runs in a thread pool to avoid blocking the async event loop
- **Batch processing** uses `asyncio.gather` for concurrent file handling
- **LLM calls** go through `httpx.AsyncClient` to any OpenAI-compatible endpoint
- **PDF handling** tries the embedded text layer first, falls back to page-by-page OCR at 300 DPI

## Project Structure

```
app.py            — FastAPI routes, OCR pipeline, file handling
processors.py     — Document classification, extraction prompts, LLM client
templates/
  index.html      — Web UI with drag-and-drop upload
requirements.txt  — Python dependencies
.env              — API keys and config (not committed)
```

## License

MIT
