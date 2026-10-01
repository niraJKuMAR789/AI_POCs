from collections import Counter
from typing import Any

import pytest
from tests.conftest import ROOT

from aegis.models import Claim, Severity
from aegis.npi import is_valid_npi, make_npi
from aegis.reference import Reference
from aegis.rules.engine import RULES, RulesEngine


@pytest.fixture
def engine(settings) -> RulesEngine:  # type: ignore[no-untyped-def]
    return RulesEngine(Reference.load(settings.reference, settings.registry))


def test_npi_check_digit() -> None:
    assert is_valid_npi("1234567893")  # CMS published example
    assert not is_valid_npi("1234567890")
    assert not is_valid_npi("123456789")
    assert is_valid_npi(make_npi("198765432"))


def test_every_rule_has_a_policy_section(settings) -> None:  # type: ignore[no-untyped-def]
    manual = (ROOT / "config" / "policy_manual.md").read_text()
    assert len(RULES) == 12
    import re

    cited = set(re.findall(r'"(AEG-\d+\.\d+)"', (ROOT / "src/aegis/rules/engine.py").read_text()))
    assert cited and all(f"### {sid}" in manual for sid in cited)


def test_engine_matches_every_injected_anomaly(engine: RulesEngine, labeled: list[dict[str, Any]]) -> None:
    mismatches = Counter()
    for row in labeled:
        fired = {f.rule_id for f in engine.evaluate(Claim.model_validate(row["claim"]))}
        if fired != set(row["expected_rules"]):
            mismatches[row["scenario"]] += 1
    assert not mismatches


def test_emergency_changes_auth_severity(engine: RulesEngine, labeled: list[dict[str, Any]]) -> None:
    from tests.conftest import pick

    office = pick(labeled, "missing_prior_auth")
    er = office.model_copy(update={"place_of_service": "23"})
    sev = lambda c: {f.severity for f in engine.evaluate(c) if f.rule_id == "R004"}  # noqa: E731
    assert sev(office) == {Severity.HARD}
    assert sev(er) == {Severity.SOFT}


def test_modifier_turns_bundling_edit_soft(engine: RulesEngine, labeled: list[dict[str, Any]]) -> None:
    from tests.conftest import pick

    claim = next(
        pick(labeled, s)
        for s in ("modifier_59_supported", "modifier_59_unsupported")
        if any(r["scenario"] == s for r in labeled)
    )
    without = claim.model_copy(deep=True)
    without.lines[1].modifiers = []
    get = lambda c: next(f for f in engine.evaluate(c) if f.rule_id == "R006")  # noqa: E731
    assert get(claim).severity == Severity.SOFT and get(claim).reviewer == "coding"
    assert get(without).severity == Severity.HARD
