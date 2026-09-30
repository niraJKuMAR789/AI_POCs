from atlas.evals.metrics import abstained, reciprocal_rank, source_recall
from atlas.evals.runner import EvalRow, render_report, summarize


def test_retrieval_metrics() -> None:
    assert source_recall(["a", "b"], ["b", "c"]) == 0.5
    assert reciprocal_rank(["x", "b"], ["b"]) == 0.5
    assert source_recall(["a"], []) is None


def test_abstention_detection() -> None:
    assert abstained("The knowledge base does not contain enough information to answer this question.")
    assert not abstained("Kestrel is certified to ISO 3691-4.")


def test_summary_and_report() -> None:
    rows = [
        EvalRow(
            id="q1",
            type="multi-hop",
            system=s,
            answer="a",
            correctness=c,
            faithfulness=1.0,
            recall=1.0,
            mrr=1.0,
            abstained=False,
            latency_s=1.0,
        )
        for s, c in (("naive-rag", 0.5), ("atlas", 1.0))
    ]
    summary = summarize(rows)
    assert summary["atlas"]["multi_hop_correctness"] == 1.0
    assert "| correctness | 0.5 | 1.0 |" in render_report(rows, summary, {"provider": "offline"})
