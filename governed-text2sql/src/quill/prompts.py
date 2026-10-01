from __future__ import annotations

from quill.data.questions import Example
from quill.llm.base import Message

SQL_SYSTEM = """You are an expert analytics engineer writing {dialect} SQL for a governed data warehouse.
Rules:
- Return exactly one read-only SELECT query (or WITH ... SELECT) inside a ```sql code block. No other statements.
- Use only the tables and columns in <schema>. Columns not listed are not available to this user; never guess them.
- Prefer the business metric definitions in <metrics> when the question mentions them.
- Dates are TEXT 'YYYY-MM-DD': use strftime('%Y', col) for years and julianday() for day differences.
- Use explicit JOIN ... ON with table aliases. Give computed columns readable aliases.
- Do not add a LIMIT unless the question asks for a top-N or a single best/worst row.
- If the question cannot be answered with this schema, return: SELECT 'UNANSWERABLE' AS reason"""


def sql_messages(
    question: str,
    *,
    dialect: str,
    schema_ddl: str,
    metrics: dict[str, str],
    examples: list[Example],
    history: list[Message] | None = None,
) -> list[Message]:
    metric_block = "\n".join(f"- {name}: {sql}" for name, sql in metrics.items()) or "(none)"
    shots = "\n\n".join(f"Q: {e.question}\n```sql\n{e.sql}\n```" for e in examples) or "(none)"
    context = "\n".join(f"{m['role']}: {m['content']}" for m in (history or [])[-4:])
    user = (
        f"<schema>\n{schema_ddl}\n</schema>\n<metrics>\n{metric_block}\n</metrics>\n"
        f"<examples>\n{shots}\n</examples>\n"
        + (f"<conversation>\n{context}\n</conversation>\n" if context else "")
        + f"<question>{question}</question>"
    )
    return [{"role": "system", "content": SQL_SYSTEM.format(dialect=dialect)}, {"role": "user", "content": user}]


def repair_message(sql: str, problem: str) -> Message:
    return {
        "role": "user",
        "content": f"The query below was rejected.\n```sql\n{sql}\n```\nProblem: {problem}\n"
        "Return a corrected query in a ```sql block that satisfies all rules.",
    }


ANSWER_SYSTEM = """You explain query results to a business user in 1-3 sentences.
Use only the numbers in <result>; do not invent values. Round money to 2 decimals and rates to percentages.
If the result is truncated or empty, say so."""


def answer_messages(question: str, sql: str, result_markdown: str, truncated: bool) -> list[Message]:
    note = " (truncated)" if truncated else ""
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {
            "role": "user",
            "content": f"<question>{question}</question>\n<sql>{sql}</sql>\n"
            f"<result{note}>\n{result_markdown}\n</result>",
        },
    ]
