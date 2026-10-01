import sqlite3

from tests.conftest import scripted_pipeline

from quill.config import Settings
from quill.pipeline import Text2SQL, extract_sql


def test_extract_sql() -> None:
    assert extract_sql("Here:\n```sql\nSELECT 1;\n```\nDone") == "SELECT 1"
    assert extract_sql("SELECT 2") == "SELECT 2"


async def test_offline_baseline_answers_seen_shape(pipeline: Text2SQL) -> None:
    result = await pipeline.ask("How many denied claims are there?")
    assert result.status == "ok"
    assert "status = 'denied'" in (result.executed_sql or "")
    assert result.rows[0][0] > 0 and result.answer


async def test_guard_violation_is_repaired(settings: Settings) -> None:
    pipe, llm = scripted_pipeline(
        settings,
        "```sql\nSELECT first_name, COUNT(*) FROM members GROUP BY first_name\n```",
        "```sql\nSELECT state, COUNT(*) AS n FROM members GROUP BY state\n```",
    )
    result = await pipe.ask("How many members per state?")
    assert [a.outcome for a in result.attempts] == ["guard_violation", "ok"]
    assert result.status == "ok" and result.columns == ["state", "n"]
    repair_prompt = llm.prompts[1][-1]["content"]
    assert "members.first_name" in repair_prompt  # the violation is fed back to the model


async def test_db_error_is_repaired(settings: Settings) -> None:
    pipe, _ = scripted_pipeline(
        settings,
        "```sql\nSELECT strftime('%Y', service_date) AS y, COUNT(*) FROM claims GROUP BY y HAVING COUNT(*) > 'x' "
        "ORDER BY nope\n```",
        "```sql\nSELECT COUNT(*) FROM claims\n```",
    )
    result = await pipe.ask("How many claims?")
    assert result.attempts[0].outcome in {"guard_violation", "db_error"}
    assert result.status == "ok"


async def test_persistent_violation_is_blocked(settings: Settings) -> None:
    pipe, _ = scripted_pipeline(settings, *["```sql\nSELECT ssn_last4 FROM members\n```"] * 3)
    result = await pipe.ask("ssn please")
    assert result.status == "blocked" and len(result.attempts) == settings.max_repairs + 1
    assert result.rows == []


async def test_prompt_never_contains_denied_columns(settings: Settings) -> None:
    pipe, llm = scripted_pipeline(settings, "```sql\nSELECT COUNT(*) FROM members\n```")
    await pipe.ask("How many members are there by name and date of birth?")
    prompt = llm.prompts[0][-1]["content"]
    assert "members" in prompt and "first_name" not in prompt and "ssn_last4" not in prompt


async def test_tenant_filter_applied(settings: Settings) -> None:
    pipe, _ = scripted_pipeline(settings, "```sql\nSELECT COUNT(*) FROM claims\n```")
    result = await pipe.ask("How many claims do I have?", role="provider_portal", context={"provider_id": 7})
    conn = sqlite3.connect(settings.db_path)
    expected = conn.execute("SELECT COUNT(*) FROM claims WHERE provider_id = 7").fetchone()[0]
    assert result.rows == [[expected]]
    assert any(r.startswith("row filter") for r in result.guard_rewrites)


async def test_unanswerable(settings: Settings) -> None:
    pipe, _ = scripted_pipeline(settings, "```sql\nSELECT 'UNANSWERABLE' AS reason\n```")
    assert (await pipe.ask("What's the weather?")).status == "unanswerable"


async def test_writes_need_confirmation(settings: Settings) -> None:
    insert = "INSERT INTO audit_flags (claim_id, reason, flagged_by) VALUES (5, 'dup billing', 'auditor')"
    pipe, _ = scripted_pipeline(settings, f"```sql\n{insert}\n```")
    result = await pipe.ask("Flag claim 5 for duplicate billing", role="claims_auditor")
    conn = sqlite3.connect(settings.db_path)
    assert result.status == "needs_confirmation"
    assert conn.execute("SELECT COUNT(*) FROM audit_flags").fetchone()[0] == 0  # nothing written yet
    assert pipe.confirm_write(result.executed_sql or "", role="claims_auditor") == 1
    assert conn.execute("SELECT reason FROM audit_flags").fetchall() == [("dup billing",)]


def test_read_connection_is_physically_read_only(pipeline: Text2SQL) -> None:
    try:
        pipeline.db.query("DELETE FROM claims")  # bypassing the guard on purpose
    except sqlite3.OperationalError as exc:
        assert "readonly" in str(exc)
    else:
        raise AssertionError("write succeeded on read connection")


def test_run_sql_applies_guard(pipeline: Text2SQL) -> None:
    blocked = pipeline.run_sql("SELECT last_name FROM members", role="analyst")
    assert blocked.status == "blocked" and "last_name" in blocked.answer
    ok = pipeline.run_sql("SELECT COUNT(*) FROM providers", role="analyst")
    assert ok.status == "ok" and ok.rows[0][0] == 60
