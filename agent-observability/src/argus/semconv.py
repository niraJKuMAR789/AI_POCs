"""Attribute names. `gen_ai.*` follow the OpenTelemetry GenAI semantic conventions; `argus.*` are ours."""

# OpenTelemetry GenAI semantic conventions
OPERATION = "gen_ai.operation.name"  # chat | embeddings | execute_tool | invoke_agent | retrieval
SYSTEM = "gen_ai.system"  # provider, e.g. "nvidia_nim", "openai"
REQUEST_MODEL = "gen_ai.request.model"
RESPONSE_MODEL = "gen_ai.response.model"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
AGENT_NAME = "gen_ai.agent.name"
CONVERSATION_ID = "gen_ai.conversation.id"
ERROR_TYPE = "error.type"
SERVICE_NAME = "service.name"
DEPLOYMENT_ENV = "deployment.environment"

# Argus extensions
SPAN_KIND = "argus.span.kind"  # agent | llm | tool | retrieval | chain
INPUT = "argus.input"  # prompt / tool arguments (content; subject to capture + redaction policy)
OUTPUT = "argus.output"  # completion / tool result
TOOL_ARGS_HASH = "argus.tool.args_hash"
AGENT_VERSION = "argus.agent.version"
PROMPT_VERSION = "argus.prompt.version"
USER_ID = "argus.user.id"
RETRIEVED_DOCS = "argus.retrieval.documents"

KINDS = ("agent", "llm", "tool", "retrieval", "chain")
OPERATION_TO_KIND = {
    "chat": "llm",
    "text_completion": "llm",
    "generate_content": "llm",
    "embeddings": "llm",
    "execute_tool": "tool",
    "invoke_agent": "agent",
    "create_agent": "agent",
    "retrieval": "retrieval",
}
