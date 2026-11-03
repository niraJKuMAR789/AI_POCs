# AI Engineering Portfolio: Niraj Kumar Marepally

AI Engineer (3+ years) building production **LLM, RAG and agentic systems**, including MCP servers, LangGraph
agents and ML platforms, for finance, healthcare and real-estate clients.
Claude Certified Architect · Databricks ML Associate · Azure AI Associate ·
[LinkedIn](https://www.linkedin.com/in/niraj-kumar-marepally)

Every project here is a self-contained, production-style codebase with typed Python, tests that run in CI
without API keys, Docker, evaluation harnesses and architecture docs. Models are served through
**NVIDIA NIM** (API Catalog or self-hosted).

## Projects

| Project | What it shows | Stack |
|---|---|---|
| [**Atlas: Adaptive GraphRAG Research Agent**](adaptive-graphrag) | Multi-hop and corpus-wide QA over private documents. Adaptive routing, hybrid + knowledge-graph retrieval, CRAG grading, Self-RAG reflection, streaming, conversation memory, an MCP server, and an eval harness benchmarked against naive RAG | LangGraph · NVIDIA Nemotron + NeMo Retriever · Qdrant · Neo4j · MCP · FastAPI · Streamlit |
| [**Quill: Governed Text-to-SQL Agent**](governed-text2sql) | NL analytics over a regulated claims warehouse. AST-level guardrails (RBAC, PII column protection, row-level security, self-repair), an Agno agent with human-approved writes, a role-bound MCP server, a red-team leak suite, and a LoRA/QLoRA fine-tuning pipeline scored by execution accuracy on held-out query shapes | Agno · MCP · NVIDIA NIM · sqlglot · TRL/PEFT · FastAPI · Streamlit |
| [**Aegis: Multi-Agent Medical Claims Auditor**](claims-audit-agents) | Deterministic rules engine + parallel LLM specialist agents (clinical, coding) whose proposals must pass evidence and citation validation, a human-review pause that survives restarts, vision intake of claim forms, a hash-chained audit ledger, and a three-way benchmark (rules-only / rules+agents / LLM-only) that measures overpayment in dollars | LangGraph (Send, interrupt, SQLite checkpoints) · NVIDIA Nemotron + Nemotron VL · MCP · FastAPI · Streamlit |
| [**Argus: Observability and Governance for Agentic AI**](agent-observability) | Self-hosted tracing for LLM agents (own SDK + OTLP/GenAI-semconv collector, OpenAI-compatible and LangGraph auto-instrumentation), PII/secret redaction and policy enforcement before storage, statistical anomaly detection grouped into incidents, and an investigator agent on NIM whose root-cause claims must cite real evidence. Benchmarked across incident intensities | OpenTelemetry · NVIDIA NIM · MCP · FastAPI · Streamlit/Plotly |

## Engineering standards across projects

- **Evaluation first:** golden datasets, LLM-as-judge plus exact metrics, and baselines to beat
- **Provider-agnostic cores:** the agent depends on protocols, so vendor SDKs stay at the edges
- **Offline-deterministic CI:** full agent paths are tested without network or API keys
- **Production details:** retries and backoff, structured outputs with fallbacks, SSE streaming,
  auth, structured logging, health checks, non-root containers
