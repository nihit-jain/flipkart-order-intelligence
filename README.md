@"
# Flipkart Order Intelligence

An end-to-end AI/ML order intelligence system combining:

- Random Forest return-risk prediction
- FashionMNIST CNN product classification
- FAISS policy retrieval
- Groundedness checking
- LangGraph agent routing
- Prompt-injection protection
- Deterministic Mock LLM response layer
- Multi-turn session memory
- FastAPI backend
- React/Vite frontend
- Retrieval and transcript evaluation

## Current Evaluation

- Precision@3: 0.63
- Recall@3: 1.00
- Representative transcript tests: 9/9 passed

## Main Components

- `agent/` — agent, RAG, tools, memory and evaluation
- `data/` — policy index and sample images
- `frontend/` — React/Vite interface
- `api.py` — FastAPI application
- `models/` — trained ML models

## Run Backend

```powershell
python -m uvicorn api:app --reload