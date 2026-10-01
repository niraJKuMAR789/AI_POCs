import json
from pathlib import Path

from tests.conftest import ROOT

from quill.data.questions import HELD_OUT_TEMPLATES, TEMPLATES
from quill.evals.execution import has_top_level_order, results_match
from quill.semantic.layer import SemanticLayer
from quill.semantic.linking import link_schema

LAYER = SemanticLayer.load(ROOT / "config" / "semantic_layer.yaml")


def test_linking_picks_tables_from_synonyms_and_values() -> None:
    link = link_schema(LAYER, "What is the denial rate for out_of_network doctors?")
    assert link.tables[:2] == ["claims", "providers"] or set(link.tables[:2]) == {"claims", "providers"}
    assert "denial_rate" in link.metrics


def test_linking_closes_over_join_paths() -> None:
    link = link_schema(LAYER, "Average claim billed amount by metal tier of the insurance plan")
    assert {"claims", "plans", "members"} <= set(link.tables)  # members bridges claims -> plans


def test_ddl_hides_denied_columns() -> None:
    ddl = LAYER.ddl(["members"], hidden_columns={"members.ssn_last4", "members.first_name"})
    assert "ssn_last4" not in ddl and "first_name" not in ddl and "state" in ddl


def test_splits_hold_out_whole_templates(warehouse: Path) -> None:
    rows = {
        s: [json.loads(x) for x in (warehouse / "splits" / f"{s}.jsonl").read_text().splitlines()]
        for s in ("train", "dev", "test")
    }
    train_templates = {r["template"] for r in rows["train"]}
    assert not train_templates & HELD_OUT_TEMPLATES
    assert {r["template"] for r in rows["test"]} >= HELD_OUT_TEMPLATES
    assert len({t.id for t in TEMPLATES}) == len(TEMPLATES) == 24


def test_execution_match_semantics() -> None:
    assert results_match([(1, 2.00001)], [(2.0, 1)], ordered=False)  # column order + float tolerance
    assert results_match([(1,), (2,)], [(2,), (1,)], ordered=False)
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)
    assert not results_match([(1,)], [(1,), (1,)], ordered=False)
    assert has_top_level_order("SELECT a FROM t ORDER BY a")
    assert not has_top_level_order("SELECT COUNT(*) FROM (SELECT a FROM t ORDER BY a)")
