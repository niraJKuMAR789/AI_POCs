import pytest

from quill.guard.validator import SQLGuard

PROVIDER = {"provider_id": 7}


@pytest.mark.parametrize(
    ("role", "sql"),
    [
        ("analyst", "SELECT state, COUNT(*) FROM members GROUP BY state"),
        ("analyst", "WITH d AS (SELECT * FROM claims WHERE status = 'denied') SELECT COUNT(*) FROM d"),
        (
            "analyst",
            "SELECT p.specialty, AVG(c.paid_amount) FROM claims c JOIN providers p USING (provider_id) GROUP BY 1",
        ),
        ("analyst", "SELECT status FROM claims UNION SELECT plan_type FROM plans"),
        ("claims_auditor", "SELECT m.first_name, c.claim_id FROM claims c JOIN members m USING (member_id)"),
    ],
)
def test_allowed_queries(guard: SQLGuard, policies, role: str, sql: str) -> None:  # type: ignore[no-untyped-def]
    result = guard.check(sql, policies[role])
    assert result.ok, result.violations


@pytest.mark.parametrize(
    ("role", "sql", "fragment"),
    [
        ("analyst", "SELECT first_name FROM members", "members.first_name"),
        ("analyst", "SELECT * FROM members", "members.ssn_last4"),
        ("analyst", "SELECT COUNT(*) FROM members m WHERE m.date_of_birth > '1990-01-01'", "date_of_birth"),
        ("analyst", "WITH x AS (SELECT last_name AS n FROM members) SELECT n FROM x", "members.last_name"),
        ("analyst", "SELECT status FROM claims UNION SELECT last_name FROM members", "members.last_name"),
        ("analyst", "SELECT (SELECT MAX(ssn_last4) FROM members) AS s", "members.ssn_last4"),
        ("claims_auditor", "SELECT ssn_last4 FROM members", "members.ssn_last4"),
        ("analyst", "DELETE FROM claims", "DELETE statements are not allowed"),
        ("analyst", "UPDATE claims SET paid_amount = 0", "UPDATE statements are not allowed"),
        ("analyst", "DROP TABLE claims", "DROP statements are not allowed"),
        ("analyst", "SELECT 1; DELETE FROM claims", "exactly one SQL statement"),
        ("analyst", "PRAGMA table_info(members)", "not allowed"),
        ("analyst", "ATTACH DATABASE 'x.db' AS x", "not allowed"),
        ("analyst", "SELECT load_extension('evil')", "load_extension"),
        ("analyst", "SELECT * FROM audit_flags", "may not read table(s): audit_flags"),
        ("analyst", "SELECT * FROM sqlite_master", "unknown table(s): sqlite_master"),
        ("analyst", "SELECT nonexistent FROM claims", "invalid column reference"),
        ("analyst", "SELEC oops", "not allowed"),
        ("analyst", "SELECT (1 FROM claims", "does not parse"),
        ("provider_portal", "SELECT * FROM members", "may not read table(s): members"),
        ("analyst", "INSERT INTO audit_flags (claim_id, reason, flagged_by) VALUES (1, 'r', 'x')", "may not write"),
        ("claims_auditor", "INSERT INTO claims (claim_id) VALUES (1)", "may not write to table 'claims'"),
    ],
)
def test_blocked_queries(guard: SQLGuard, policies, role: str, sql: str, fragment: str) -> None:  # type: ignore[no-untyped-def]
    context = PROVIDER if role == "provider_portal" else {}
    result = guard.check(sql, policies[role], context)
    assert not result.ok
    assert fragment in result.feedback()


def test_row_filters_are_injected_everywhere(guard: SQLGuard, policies) -> None:  # type: ignore[no-untyped-def]
    sql = (
        "SELECT c.status, SUM(l.line_paid) FROM claims c JOIN claim_lines l ON l.claim_id = c.claim_id "
        "WHERE c.claim_id IN (SELECT claim_id FROM claims WHERE paid_amount > 100) GROUP BY c.status"
    )
    result = guard.check(sql, policies["provider_portal"], PROVIDER)
    assert result.ok and result.sql
    assert result.sql.count("provider_id = 7") == 3  # outer claims, subquery claims, claim_lines filter
    assert "LIMIT 200" in result.sql


def test_row_filter_requires_context(guard: SQLGuard, policies) -> None:  # type: ignore[no-untyped-def]
    result = guard.check("SELECT COUNT(*) FROM claims", policies["provider_portal"], {})
    assert not result.ok and "requires context: provider_id" in result.feedback()


def test_context_values_cannot_inject_sql(guard: SQLGuard, policies) -> None:  # type: ignore[no-untyped-def]
    result = guard.check("SELECT COUNT(*) FROM claims", policies["provider_portal"], {"provider_id": "7 OR 1=1"})
    assert result.ok and result.sql
    assert "provider_id = '7 OR 1=1'" in result.sql  # bound as a string literal, not SQL


def test_limit_is_capped_but_smaller_limits_kept(guard: SQLGuard, policies) -> None:  # type: ignore[no-untyped-def]
    capped = guard.check("SELECT claim_id FROM claims LIMIT 50000", policies["analyst"])
    kept = guard.check("SELECT claim_id FROM claims LIMIT 5", policies["analyst"])
    assert capped.sql and capped.sql.endswith("LIMIT 1000") and capped.rewrites == ["limit capped at 1000"]
    assert kept.sql and kept.sql.endswith("LIMIT 5") and kept.rewrites == []


def test_auditor_insert_allowed(guard: SQLGuard, policies) -> None:  # type: ignore[no-untyped-def]
    sql = "INSERT INTO audit_flags (claim_id, reason, flagged_by) VALUES (12, 'upcoding', 'qa')"
    result = guard.check(sql, policies["claims_auditor"])
    assert result.ok and result.statement == "INSERT"
