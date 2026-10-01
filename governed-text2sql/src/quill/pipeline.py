"""Governed NL→SQL pipeline.

    question
      → schema linking (semantic layer), filtered to what the caller's role may see
      → few-shot retrieval from the verified example bank
      → SQL generation (NVIDIA NIM / fine-tuned model)
      → AST guard (RBAC, PII, row-level security, LIMIT)  ─┐ violation or DB error:
      → read-only execution with timeout                   ─┴→ feedback → repair (bounded)
      → grounded natural-language answer

Writes (INSERT into a role's write tables) are never auto-executed: they come back as
`needs_confirmation` and run only through `confirm_write`, which re-checks the guard.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from quill.config import Settings
from quill.data.questions import Example, load_split
from quill.db.engine import Database, QueryTimeoutError
from quill.guard.policy import RolePolicy, load_policies
from quill.guard.validator import SQLGuard
from quill.llm.base import Message
from quill.llm.factory import Models, build_models
from quill.prompts import answer_messages, repair_message, sql_messages
from quill.semantic.layer import SemanticLayer
from quill.semantic.linking import link_schema
from quill.text import jaccard

_SQL_BLOCK_RE = re.compile(r"```(?:sql)?\s*\n?(.*?)```", re.S | re.I)
LARGE_TABLES = frozenset({"claims", "claim_lines"})

Status = Literal["ok", "needs_confirmation", "unanswerable", "blocked", "error"]


class Attempt(BaseModel):
    sql: str
    outcome: Literal["ok", "guard_violation", "db_error", "timeout", "unanswerable", "write_pending"]
    feedback: str = ""


class AskResult(BaseModel):
    question: str
    role: str
    status: Status
    sql: str | None = None
    executed_sql: str | None = None
    columns: list[str] = []
    rows: list[list[Any]] = []
    row_count: int = 0
    truncated: bool = False
    answer: str = ""
    linked_tables: list[str] = []
    guard_rewrites: list[str] = []
    plan_warnings: list[str] = []
    attempts: list[Attempt] = []
    latency_ms: float = 0.0


def extract_sql(text: str) -> str:
    match = _SQL_BLOCK_RE.search(text)
    sql = match.group(1) if match else text
    return sql.strip().rstrip(";").strip()


@dataclass
class Text2SQL:
    settings: Settings
    layer: SemanticLayer
    policies: dict[str, RolePolicy]
    db: Database
    models: Models
    examples: list[Example] = field(default_factory=list)

    @classmethod
    def from_settings(cls, settings: Settings, models: Models | None = None) -> Text2SQL:
        layer = SemanticLayer.load(settings.semantic_layer)
        examples = load_split(settings.examples_path) if Path(settings.examples_path).exists() else []
        return cls(
            settings=settings,
            layer=layer,
            policies=load_policies(settings.policies),
            db=Database(settings.db_path, timeout_s=settings.query_timeout_s, large_tables=LARGE_TABLES),
            models=models or build_models(settings),
            examples=examples,
        )

    def __post_init__(self) -> None:
        self.guard = SQLGuard(self.layer.sql_schema, dialect=self.layer.dialect)

    def policy(self, role: str) -> RolePolicy:
        if role not in self.policies:
            raise KeyError(f"unknown role '{role}'")
        return self.policies[role]

    # --- prompt construction -------------------------------------------------------------

    def visible_schema(self, question: str, policy: RolePolicy) -> tuple[list[str], str, dict[str, str]]:
        """Linked tables ∩ role tables; denied columns are removed from the prompt entirely."""
        link = link_schema(self.layer, question)
        allowed = {t.lower() for t in policy.tables}
        tables = [t for t in link.tables if t in allowed] or [t for t in policy.tables if t in self.layer.tables][:4]
        ddl = self.layer.ddl(tables, hidden_columns=policy.denied)
        metrics = {m: self.layer.metrics[m].sql for m in link.metrics}
        return tables, ddl, metrics

    def few_shot(self, question: str, policy: RolePolicy) -> list[Example]:
        allowed = {t.lower() for t in policy.tables}
        usable = [e for e in self.examples if all(t in allowed for t in _tables_in(e.sql))]
        return sorted(usable, key=lambda e: jaccard(question, e.question), reverse=True)[: self.settings.few_shot_k]

    # --- main entry points -----------------------------------------------------------------

    async def ask(
        self,
        question: str,
        *,
        role: str | None = None,
        context: dict[str, Any] | None = None,
        history: list[Message] | None = None,
        summarize: bool = True,
    ) -> AskResult:
        started = time.perf_counter()
        role = role or self.settings.default_role
        policy = self.policy(role)
        context = context or {}
        tables, ddl, metrics = self.visible_schema(question, policy)
        result = AskResult(question=question, role=role, status="error", linked_tables=tables)

        messages = sql_messages(
            question,
            dialect=self.layer.dialect,
            schema_ddl=ddl,
            metrics=metrics,
            examples=self.few_shot(question, policy),
            history=history,
        )
        for _ in range(self.settings.max_repairs + 1):
            raw = await self.models.sql.complete(messages)
            sql = extract_sql(raw)
            result.sql = sql
            if "UNANSWERABLE" in sql.upper():
                result.attempts.append(Attempt(sql=sql, outcome="unanswerable"))
                result.status = "unanswerable"
                result.answer = "This question can't be answered with the data available to your role."
                break
            verdict = self.guard.check(sql, policy, context)
            if not verdict.ok:
                result.attempts.append(Attempt(sql=sql, outcome="guard_violation", feedback=verdict.feedback()))
                messages = [*messages, {"role": "assistant", "content": raw}, repair_message(sql, verdict.feedback())]
                continue
            assert verdict.sql is not None
            if verdict.statement == "INSERT":
                result.attempts.append(Attempt(sql=verdict.sql, outcome="write_pending"))
                result.status, result.executed_sql = "needs_confirmation", verdict.sql
                result.answer = "This request writes data. Review the SQL and confirm to execute it."
                break
            try:
                query = self.db.query(verdict.sql, max_rows=policy.max_rows)
            except QueryTimeoutError as exc:
                result.attempts.append(Attempt(sql=verdict.sql, outcome="timeout", feedback=str(exc)))
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    repair_message(sql, f"{exc}; write a more selective or simpler query"),
                ]
                continue
            except sqlite3.Error as exc:
                result.attempts.append(Attempt(sql=verdict.sql, outcome="db_error", feedback=str(exc)))
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    repair_message(sql, f"database error: {exc}"),
                ]
                continue
            result.attempts.append(Attempt(sql=verdict.sql, outcome="ok"))
            result.status = "ok"
            result.executed_sql = verdict.sql
            result.guard_rewrites = verdict.rewrites
            result.plan_warnings = self.db.plan_warnings(verdict.sql)
            result.columns = query.columns
            result.rows = [list(r) for r in query.rows]
            result.row_count = len(query.rows)
            result.truncated = query.truncated
            if summarize:
                result.answer = await self.models.answer.complete(
                    answer_messages(question, verdict.sql, query.to_markdown(), query.truncated)
                )
            break
        else:
            result.status = "blocked" if result.attempts[-1].outcome == "guard_violation" else "error"
            result.answer = f"Could not produce an allowed, working query: {result.attempts[-1].feedback}"
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return result

    def confirm_write(self, sql: str, *, role: str, context: dict[str, Any] | None = None) -> int:
        """Human-in-the-loop write path: re-validate, then execute the INSERT."""
        verdict = self.guard.check(sql, self.policy(role), context)
        if not verdict.ok or verdict.statement != "INSERT" or verdict.sql is None:
            raise PermissionError(verdict.feedback() or "only guarded INSERT statements can be confirmed")
        return self.db.execute_write(verdict.sql)

    def run_sql(self, sql: str, *, role: str, context: dict[str, Any] | None = None) -> AskResult:
        """Execute caller-provided SQL under the same guard (used by MCP / power users)."""
        started = time.perf_counter()
        policy = self.policy(role)
        result = AskResult(question="", role=role, status="blocked", sql=sql)
        verdict = self.guard.check(sql, policy, context)
        if not verdict.ok or verdict.sql is None:
            result.attempts.append(Attempt(sql=sql, outcome="guard_violation", feedback=verdict.feedback()))
            result.answer = verdict.feedback()
            return result
        if verdict.statement == "INSERT":
            result.status, result.executed_sql = "needs_confirmation", verdict.sql
            return result
        try:
            query = self.db.query(verdict.sql, max_rows=policy.max_rows)
        except (sqlite3.Error, QueryTimeoutError) as exc:
            result.status, result.answer = "error", str(exc)
            return result
        result.status, result.executed_sql = "ok", verdict.sql
        result.guard_rewrites = verdict.rewrites
        result.columns, result.rows = query.columns, [list(r) for r in query.rows]
        result.row_count, result.truncated = len(query.rows), query.truncated
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return result


def _tables_in(sql: str) -> set[str]:
    return {m.lower() for m in re.findall(r"\b(?:FROM|JOIN)\s+([a-z_]+)", sql, flags=re.I)}
