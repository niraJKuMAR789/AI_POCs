"""SQL guardrails enforced on the parsed AST (sqlglot), independent of the LLM.

Checks, in order:
  1. exactly one statement that parses in the target dialect
  2. statement type: SELECT/UNION/WITH for every role; INSERT only into the role's write tables
  3. dangerous functions (file access, extension loading, sleep-style DoS)
  4. table allowlist per role (CTE names excluded)
  5. column resolution against the schema (stars expanded) and the per-role denied/PII columns
Rewrites:
  6. row-level security: the role's row filters are AND-ed into every SELECT scope that reads the table
  7. LIMIT capped at the role's max_rows

Violations are returned as structured messages, which the pipeline feeds back to the LLM for repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope

from quill.guard.policy import RolePolicy

DANGEROUS_FUNCTIONS = frozenset(
    {
        "load_extension",
        "readfile",
        "writefile",
        "edit",
        "fts3_tokenizer",
        "randomblob",
        "zeroblob",
        "pg_sleep",
        "pg_read_file",
        "pg_ls_dir",
        "lo_import",
        "lo_export",
        "dblink",
        "sleep",
        "benchmark",
    }
)


@dataclass
class GuardResult:
    ok: bool
    sql: str | None = None
    statement: str = ""
    violations: list[str] = field(default_factory=list)
    rewrites: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)

    def feedback(self) -> str:
        return "; ".join(self.violations)


class SQLGuard:
    def __init__(self, schema: dict[str, dict[str, str]], dialect: str = "sqlite") -> None:
        self.schema = {t.lower(): {c.lower(): ty for c, ty in cols.items()} for t, cols in schema.items()}
        self.dialect = dialect
        self._qualify_schema: dict[str, object] = dict(self.schema)

    def check(self, sql: str, policy: RolePolicy, context: dict[str, Any] | None = None) -> GuardResult:
        context = context or {}
        missing = [k for k in policy.required_context if k not in context]
        if missing:
            return GuardResult(ok=False, violations=[f"role '{policy.name}' requires context: {', '.join(missing)}"])
        try:
            statements = [s for s in sqlglot.parse(sql, read=self.dialect) if s is not None]
        except ParseError as exc:
            return GuardResult(ok=False, violations=[f"SQL does not parse: {_first_line(exc)}"])
        if len(statements) != 1:
            return GuardResult(ok=False, violations=["exactly one SQL statement is allowed"])
        tree = statements[0]

        if isinstance(tree, exp.Insert):
            return self._check_insert(tree, policy)
        if not isinstance(tree, exp.Query):
            kind = tree.key.upper()
            return GuardResult(ok=False, statement=kind, violations=[f"{kind} statements are not allowed"])
        return self._check_query(tree, policy, context)

    # --- SELECT --------------------------------------------------------------------------

    def _check_query(self, tree: exp.Query, policy: RolePolicy, context: dict[str, Any]) -> GuardResult:
        result = GuardResult(ok=False, statement="SELECT")
        if bad := self._dangerous_functions(tree):
            result.violations.append(f"function(s) not allowed: {', '.join(sorted(bad))}")
            return result

        cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
        tables = sorted({t.name.lower() for t in tree.find_all(exp.Table)} - cte_names)
        result.tables = tables
        unknown = [t for t in tables if t not in self.schema]
        if unknown:
            result.violations.append(f"unknown table(s): {', '.join(unknown)}")
            return result
        forbidden = [t for t in tables if t not in {x.lower() for x in policy.tables}]
        if forbidden:
            result.violations.append(f"role '{policy.name}' may not read table(s): {', '.join(forbidden)}")
            return result

        try:
            qualified = qualify(
                tree.copy(),
                schema=self._qualify_schema,
                dialect=self.dialect,
                validate_qualify_columns=True,
                quote_identifiers=False,
            )
        except OptimizeError as exc:
            result.violations.append(f"invalid column reference: {_first_line(exc)}")
            return result

        denied = self._denied_columns(qualified, policy)
        if denied:
            result.violations.append(
                f"role '{policy.name}' may not access column(s): {', '.join(sorted(denied))}. "
                "Use aggregates or non-identifying columns instead."
            )
            return result

        rewritten = tree.copy()
        result.rewrites += self._apply_row_filters(rewritten, policy, context)
        result.rewrites += self._cap_limit(rewritten, policy.max_rows)
        result.ok = True
        result.sql = rewritten.sql(dialect=self.dialect)
        return result

    def _denied_columns(self, qualified: exp.Expr, policy: RolePolicy) -> set[str]:
        denied = policy.denied
        if not denied:
            return set()
        hits: set[str] = set()
        for scope in traverse_scope(qualified):
            for column in scope.columns:
                source = scope.sources.get(column.table)
                if isinstance(source, exp.Table):
                    ref = f"{source.name.lower()}.{column.name.lower()}"
                    if ref in denied:
                        hits.add(ref)
        return hits

    def _apply_row_filters(self, tree: exp.Expr, policy: RolePolicy, context: dict[str, Any]) -> list[str]:
        if not policy.row_filters:
            return []
        applied = []
        for scope in traverse_scope(tree):
            select = scope.expression
            if not isinstance(select, exp.Select):
                continue
            for alias, source in _table_sources(scope):
                template = policy.row_filters.get(source.name.lower())
                if template is None:
                    continue
                condition = self._render_filter(template, alias, context)
                select.where(condition, append=True, copy=False, dialect=self.dialect)
                applied.append(f"row filter on {source.name}: {condition}")
        return applied

    def _render_filter(self, template: str, alias: str, context: dict[str, Any]) -> str:
        values = {k: _sql_literal(v) for k, v in context.items()}
        predicate = sqlglot.parse_one(template.format(**values), read=self.dialect)
        for column in predicate.find_all(exp.Column):
            if column.find_ancestor(exp.Select) is None:  # leave columns inside subqueries alone
                column.set("table", exp.to_identifier(alias))
        return predicate.sql(dialect=self.dialect)

    @staticmethod
    def _cap_limit(tree: exp.Expr, max_rows: int) -> list[str]:
        if not isinstance(tree, exp.Query):
            return []
        limit = tree.args.get("limit")
        current = None
        if limit is not None and isinstance(limit.expression, exp.Literal) and limit.expression.is_int:
            current = int(limit.expression.this)
        if current is not None and current <= max_rows:
            return []
        tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
        return [f"limit capped at {max_rows}"]

    # --- INSERT --------------------------------------------------------------------------

    def _check_insert(self, tree: exp.Insert, policy: RolePolicy) -> GuardResult:
        target = tree.find(exp.Table)
        name = target.name.lower() if target else ""
        result = GuardResult(ok=False, statement="INSERT", tables=[name])
        if name not in {t.lower() for t in policy.write_tables}:
            result.violations.append(f"role '{policy.name}' may not write to table '{name}'")
            return result
        source = tree.expression
        if isinstance(source, exp.Query):
            inner = self._check_query(source, policy, {})
            if not inner.ok:
                result.violations += inner.violations
                return result
        elif not isinstance(source, exp.Values):
            result.violations.append("INSERT must use VALUES or a SELECT")
            return result
        if bad := self._dangerous_functions(tree):
            result.violations.append(f"function(s) not allowed: {', '.join(sorted(bad))}")
            return result
        result.ok = True
        result.sql = tree.sql(dialect=self.dialect)
        return result

    @staticmethod
    def _dangerous_functions(tree: exp.Expr) -> set[str]:
        names = set()
        for func in tree.find_all(exp.Func):
            name = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()
            if name in DANGEROUS_FUNCTIONS:
                names.add(name)
        return names


def _table_sources(scope: Scope) -> list[tuple[str, exp.Table]]:
    return [(alias, src) for alias, src in scope.sources.items() if isinstance(src, exp.Table)]


def _sql_literal(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"unsupported context value: {value!r}")
    if isinstance(value, int | float):
        return str(value)
    return "'" + value.replace("'", "''") + "'"


def _first_line(exc: Exception) -> str:
    return str(exc).strip().splitlines()[0][:300]
