from atlas.models import normalize_entity_name
from atlas.retrieval.fusion import reciprocal_rank_fusion
from atlas.text import overlap_score, split_sentences, tokenize


def test_tokenize_drops_stopwords_and_stems() -> None:
    assert tokenize("The robots are dispatching missions across sites") == [
        "robot",
        "dispatching",
        "mission",
        "across",
        "site",
    ]


def test_split_sentences() -> None:
    assert split_sentences("First one. Second one! Third?") == ["First one.", "Second one!", "Third?"]


def test_overlap_score_bounds() -> None:
    assert overlap_score("kafka retention", "Kafka retention is 72 hours") == 1.0
    assert overlap_score("kafka retention", "unrelated text") == 0.0


def test_normalize_entity_name() -> None:
    assert normalize_entity_name("  NVIDIA  Corp. ") == normalize_entity_name("nvidia corp") == "nvidia corp"


def test_rrf_rewards_agreement() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "a"], ["b"]]))
    assert max(fused, key=fused.__getitem__) == "b"
    assert fused["c"] < fused["a"]
