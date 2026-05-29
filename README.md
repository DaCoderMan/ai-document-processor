# Ai Document Processor

AI-powered document processing pipeline — PDF/image upload, OCR, and intelligent data extraction via FastAPI

## Overview

- Repository: `DaCoderMan/ai-document-processor`
- Primary language: `Python`
- Project type: FastAPI service
- Visibility: public
- Default branch: `main`

This README was generated after a static review of the repository structure, package files, scripts, source files, and existing documentation.

## What this project contains

- API/server endpoints or route handlers are present.
- Python modules include reusable classes/functions for automation or service logic.

## Repository profile

- Files scanned: 9
- Estimated source/config lines reviewed: 653
- Top-level directories: `templates`
- File-type mix: [none]: 3, .py: 2, .example: 1, .txt: 1, .md: 1, .html: 1

## Key files and folders

- `Dockerfile`
- `LICENSE`
- `README.md`
- `requirements.txt`

## Python code map

API route decorators:

- `app.py: @app.get("/", response_class=HTMLResponse)`
- `app.py: @app.post("/api/process")`
- `app.py: @app.post("/api/batch")`
- `app.py: @app.get("/health")`

Classes sampled:

- `processors.py:DocumentType`

Functions sampled:

- `app.py:extract_text_from_image`
- `app.py:extract_text_from_pdf`
- `app.py:run_ocr`
- `app.py:index`
- `app.py:process_document`
- `app.py:batch_process`
- `app.py:health`
- `app.py:_process_one`
- `processors.py:build_classification_prompt`
- `processors.py:build_extraction_prompt`
- `processors.py:call_llm`
- `processors.py:classify_document`
- `processors.py:parse_llm_json`
- `processors.py:extract_fields`

## Getting started

```bash
git clone https://github.com/DaCoderMan/ai-document-processor
cd ai-document-processor
```

Common commands inferred from the repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration clues

Environment/configuration names referenced in the codebase include:

- `API`
- `API_BASE`
- `API_KEY`
- `LLM_API_BASE`
- `LLM_API_KEY`

Create a local `.env` file only if the project expects one, and never commit real secrets.

## Security and maintenance notes

No obvious hardcoded secret patterns were detected during this lightweight static pass. This is not a full security audit.

- Keep README setup instructions aligned with actual scripts and deployment steps.
- Document required environment variables in `.env.example` rather than committing real values.
- Run the project-specific test/build command before merging future code changes.

## Generated documentation note

This README was prepared by Hermes Agent from repository analysis. Review the wording and project-specific assumptions before merging.
