"""
Document type detection and structured field extraction via LLM.
Supports invoices, contracts, receipts, and generic documents.
"""

import json
import logging
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    INVOICE = "invoice"
    CONTRACT = "contract"
    RECEIPT = "receipt"
    GENERIC = "generic"


# Extraction schemas per document type — sent to the LLM as instructions.
EXTRACTION_SCHEMAS = {
    DocumentType.INVOICE: {
        "document_type": "invoice",
        "fields": {
            "invoice_number": "string or null",
            "date": "ISO date string or null",
            "due_date": "ISO date string or null",
            "vendor": {"name": "string", "address": "string or null"},
            "bill_to": {"name": "string or null", "address": "string or null"},
            "currency": "3-letter code or null",
            "subtotal": "number or null",
            "tax": "number or null",
            "total": "number or null",
            "line_items": [
                {
                    "description": "string",
                    "quantity": "number or null",
                    "unit_price": "number or null",
                    "amount": "number or null",
                }
            ],
        },
    },
    DocumentType.CONTRACT: {
        "document_type": "contract",
        "fields": {
            "title": "string or null",
            "date": "ISO date string or null",
            "effective_date": "ISO date string or null",
            "expiration_date": "ISO date string or null",
            "parties": [{"name": "string", "role": "string or null"}],
            "key_terms": ["string — short summary of each significant clause"],
            "governing_law": "jurisdiction string or null",
            "signatures": [{"name": "string or null", "date": "string or null"}],
        },
    },
    DocumentType.RECEIPT: {
        "document_type": "receipt",
        "fields": {
            "store_name": "string or null",
            "store_address": "string or null",
            "date": "ISO date string or null",
            "items": [
                {"description": "string", "quantity": "number or null", "price": "number or null"}
            ],
            "subtotal": "number or null",
            "tax": "number or null",
            "total": "number or null",
            "payment_method": "string or null",
        },
    },
    DocumentType.GENERIC: {
        "document_type": "generic",
        "fields": {
            "title": "string or null",
            "date": "string or null",
            "summary": "Brief summary of the document content",
            "key_entities": ["People, organizations, or places mentioned"],
            "key_values": {"field_name": "value — any important data points found"},
        },
    },
}


def build_classification_prompt(text: str) -> str:
    """Build a prompt that asks the LLM to classify document type."""
    return (
        "Classify the following document text into exactly one category: "
        "invoice, contract, receipt, or generic.\n"
        "Respond with ONLY the category name in lowercase, nothing else.\n\n"
        f"--- DOCUMENT TEXT ---\n{text[:3000]}"
    )


def build_extraction_prompt(text: str, doc_type: DocumentType) -> str:
    """Build a prompt that asks the LLM to extract structured fields."""
    schema = EXTRACTION_SCHEMAS[doc_type]
    return (
        f"Extract structured data from this {doc_type.value} document.\n"
        f"Return valid JSON matching this schema:\n"
        f"```json\n{json.dumps(schema, indent=2)}\n```\n\n"
        "Rules:\n"
        "- Use null for fields you cannot find.\n"
        "- Dates should be ISO 8601 (YYYY-MM-DD) when possible.\n"
        "- Numbers should be plain numbers, not strings.\n"
        "- Do NOT invent data. Only extract what is present.\n"
        "- Return ONLY the JSON object, no markdown fences or commentary.\n\n"
        f"--- DOCUMENT TEXT ---\n{text[:6000]}"
    )


async def call_llm(
    prompt: str,
    *,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> str:
    """Call an OpenAI-compatible chat completions endpoint."""
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 4096,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


async def classify_document(
    text: str, *, api_base: str, api_key: str, model: str
) -> DocumentType:
    """Use the LLM to determine what kind of document this is."""
    prompt = build_classification_prompt(text)
    raw = await call_llm(prompt, api_base=api_base, api_key=api_key, model=model)
    raw = raw.lower().strip().strip('"').strip("'")

    for dt in DocumentType:
        if dt.value in raw:
            return dt
    logger.warning("LLM returned unrecognized type '%s', falling back to generic", raw)
    return DocumentType.GENERIC


def parse_llm_json(raw: str) -> dict:
    """Best-effort parse of JSON from LLM output, handling markdown fences."""
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    return json.loads(text)


async def extract_fields(
    text: str,
    doc_type: DocumentType | None = None,
    *,
    api_base: str,
    api_key: str,
    model: str,
) -> dict:
    """
    Full extraction pipeline: classify (if needed) then extract structured fields.
    Returns a dict with document_type, fields, and metadata.
    """
    if not text or not text.strip():
        return {
            "document_type": "unknown",
            "fields": {},
            "error": "No text content to process",
        }

    # Step 1: classify
    if doc_type is None:
        doc_type = await classify_document(text, api_base=api_base, api_key=api_key, model=model)

    # Step 2: extract
    prompt = build_extraction_prompt(text, doc_type)
    raw = await call_llm(prompt, api_base=api_base, api_key=api_key, model=model)

    try:
        fields = parse_llm_json(raw)
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM JSON output: %s", raw[:500])
        return {
            "document_type": doc_type.value,
            "fields": {},
            "raw_response": raw[:2000],
            "error": "LLM returned invalid JSON",
        }

    return {
        "document_type": doc_type.value,
        "fields": fields.get("fields", fields),
    }
