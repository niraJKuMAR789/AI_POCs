"""Semantic layer: business metadata over the physical schema."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Column(BaseModel):
    description: str = ""
    pii: bool = False
    restricted: bool = False
    values: list[str] = Field(default_factory=list)
    references: str | None = None


class Table(BaseModel):
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    columns: dict[str, Column]


class Metric(BaseModel):
    description: str
    sql: str
    synonyms: list[str] = Field(default_factory=list)


class SemanticLayer(BaseModel):
    name: str
    description: str = ""
    dialect: str = "sqlite"
    tables: dict[str, Table]
    metrics: dict[str, Metric] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> SemanticLayer:
        return cls.model_validate(yaml.safe_load(path.read_text()))

    @cached_property
    def sql_schema(self) -> dict[str, dict[str, str]]:
        """sqlglot-style schema mapping used to qualify and validate SQL."""
        return {t: dict.fromkeys(table.columns, "TEXT") for t, table in self.tables.items()}

    def joins(self) -> list[tuple[str, str]]:
        """Foreign-key edges as ("table.column", "table.column")."""
        return [
            (f"{t}.{c}", col.references)
            for t, table in self.tables.items()
            for c, col in table.columns.items()
            if col.references
        ]

    def ddl(self, tables: list[str] | None = None, hidden_columns: set[str] | None = None) -> str:
        """Annotated, prompt-friendly DDL for the given tables (hidden columns are omitted)."""
        hidden = hidden_columns or set()
        blocks = []
        for name in tables or list(self.tables):
            table = self.tables[name]
            cols = []
            for col_name, col in table.columns.items():
                if f"{name}.{col_name}" in hidden:
                    continue
                notes = [col.description]
                if col.values:
                    notes.append("values: " + ", ".join(col.values))
                if col.references:
                    notes.append(f"FK -> {col.references}")
                cols.append(f"  {col_name}  -- {'; '.join(n for n in notes if n)}")
            blocks.append(f"-- {table.description}\nTABLE {name} (\n" + ",\n".join(cols) + "\n)")
        return "\n\n".join(blocks)
