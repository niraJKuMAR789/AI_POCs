# AI Engineering Portfolio: Niraj Kumar Marepally

AI Engineer (5+ years) building production **LLM, RAG and agentic systems**, including MCP servers, LangGraph
agents and ML platforms, for finance, healthcare and real-estate clients.
Claude Certified Architect · Databricks ML Associate · Azure AI Associate ·
[LinkedIn](https://www.linkedin.com/in/niraj-marepally)

Every project here is a self-contained, production-style codebase with typed Python, tests that run in CI
without API keys, Docker, evaluation harnesses and architecture docs. Models are served through
**NVIDIA NIM** (API Catalog or self-hosted).

## Projects

| Project | What it shows | Stack |
|---|---|---|
| [**Atlas: Adaptive GraphRAG Research Agent**](adaptive-graphrag) | Multi-hop and corpus-wide QA over private documents. Adaptive routing, hybrid + knowledge-graph retrieval, CRAG grading, Self-RAG reflection, streaming, conversation memory, an MCP server, and an eval harness benchmarked against naive RAG | LangGraph · NVIDIA Nemotron + NeMo Retriever · Qdrant · Neo4j · MCP · FastAPI · Streamlit |
| **Text-to-SQL agent + fine-tuning** *(in progress)* | Secure NL→SQL agent with a semantic layer, SQL guardrails and RBAC, plus a LoRA fine-tuning and evaluation pipeline on BIRD/Spider | Agno · MCP · NVIDIA NIM · PostgreSQL · PEFT/LoRA |

## Engineering standards across projects

- **Evaluation first:** golden datasets, LLM-as-judge plus exact metrics, and baselines to beat
- **Provider-agnostic cores:** the agent depends on protocols, so vendor SDKs stay at the edges
- **Offline-deterministic CI:** full agent paths are tested without network or API keys
- **Production details:** retries and backoff, structured outputs with fallbacks, SSE streaming,
  auth, structured logging, health checks, non-root containers
