# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**echo-hint** is a Python project focused on semantic search and retrieval using FlagEmbedding (BGE models). The project demonstrates how to build vector-based retrieval systems using Chinese language models for document similarity search.

## Development Commands

### Environment Setup
```bash
# The project uses uv for dependency management
# Virtual environment is in .venv/

# Install dependencies
uv pip install -e .

# Activate virtual environment
source .venv/bin/activate
```

### Running the Demo
```bash
# Run the quick retrieval demo (interactive semantic search)
python scripts/quick_retrieval_demo.py
```

## Architecture

### Core Dependencies
- **FlagEmbedding**: Primary library for text embedding and retrieval
  - Uses BGE (BAAI General Embedding) models
  - Default model: `BAAI/bge-small-zh-v1.5` (Chinese, optimized for speed)
  - FP16 precision enabled for efficiency

### Project Structure
- `scripts/quick_retrieval_demo.py`: Interactive retrieval system demonstration
  - Loads BGE embedding model
  - Encodes document corpus into vector embeddings
  - Performs cosine similarity-based retrieval
  - Includes both batch query examples and interactive mode
  - Bilingual (Chinese/English) documentation and output

### Key Technical Details

**Retrieval Pipeline:**
1. Documents are encoded into embeddings once at startup
2. Query text is encoded at search time
3. Cosine similarity computed between query embedding and all document embeddings
4. Results ranked by similarity score, top-k returned

**Model Loading:**
- Models are cached locally after first download (Hugging Face cache)
- `.cache/`, `*.model`, `*.pt`, `*.bin`, `*.onnx` are gitignored to prevent committing large model files

**Language Focus:**
- Primary target: Chinese language text
- Uses specialized Chinese embedding model
- Demo includes Chinese city descriptions and technical concepts
