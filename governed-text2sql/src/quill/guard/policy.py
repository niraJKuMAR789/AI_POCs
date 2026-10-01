"""Role policies loaded from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class RolePolicy(BaseModel):
    name: str
    description: str = ""
    tables: list[str]
    deny_columns: list[str] = Field(default_factory=list)
    write_tables: list[str] = Field(default_factory=list)
    row_filters: dict[str, str] = Field(default_factory=dict)
    required_context: list[str] = Field(default_factory=list)
    max_rows: int = 1000

    @property
    def denied(self) -> set[str]:
        return {c.lower() for c in self.deny_columns}


def load_policies(path: Path) -> dict[str, RolePolicy]:
    raw = yaml.safe_load(path.read_text())["roles"]
    return {name: RolePolicy(name=name, **spec) for name, spec in raw.items()}
