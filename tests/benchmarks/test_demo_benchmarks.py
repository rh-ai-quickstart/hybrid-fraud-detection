"""Stage 4: lightweight performance guards for the self-contained demo path."""

import pathlib
import sys
import time

SRC_DIR = pathlib.Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

from scorer import DemoLLMScorer, HybridScorer, RuleEngine, TransactionRequest


def _scorer() -> HybridScorer:
    return HybridScorer(
        rule_engine=RuleEngine(),
        llm_scorer=DemoLLMScorer(model="benchmark-model"),
        model_name="benchmark-model",
    )


def test_rule_only_scoring_is_below_10ms():
    scorer = _scorer()
    tx = TransactionRequest(amount=50, country="US", category="retail")

    start = time.perf_counter()
    result = scorer.score(tx)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert result.llm_skipped is True
    assert elapsed_ms < 10


def test_demo_model_scoring_is_below_500ms():
    scorer = _scorer()
    tx = TransactionRequest(amount=5000, country="US", category="retail")

    start = time.perf_counter()
    result = scorer.score(tx)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert result.llm_skipped is False
    assert result.model == "demo-simulator"
    assert elapsed_ms < 500


def test_demo_batch_of_100_is_below_five_seconds():
    scorer = _scorer()
    transactions = [
        TransactionRequest(amount=5000, country="US", category="retail")
        for _ in range(100)
    ]

    start = time.perf_counter()
    results = [scorer.score(tx) for tx in transactions]
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert len(results) == 100
    assert elapsed_ms < 5000
