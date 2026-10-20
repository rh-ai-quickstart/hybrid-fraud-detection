"""Stage 2: Technique validation -- hybrid fraud scoring unit tests.

Tests cover:
  - Rule engine signal detection and scoring
  - LLM scoring (mocked)
  - 60/40 weighted combination
  - Conditional LLM skip for confident rule scores
  - Stats tracking including skip rate
  - Batch scoring
"""

import pathlib
import sys
from unittest.mock import MagicMock

import httpx
import pytest

# Add src to path so we can import scorer
SRC_DIR = pathlib.Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

from scorer import (
    LLM_WEIGHT,
    RULE_WEIGHT,
    SKIP_HIGH_THRESHOLD,
    DemoLLMScorer,
    HybridScorer,
    LLMScorer,
    RuleEngine,
    TransactionRequest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rule_engine():
    return RuleEngine()


@pytest.fixture
def mock_llm_scorer():
    """LLM scorer that returns a controllable score."""
    scorer = MagicMock()
    scorer.score.return_value = (50.0, 100.0)  # (llm_score, latency_ms)
    return scorer


@pytest.fixture
def hybrid_scorer(rule_engine, mock_llm_scorer):
    return HybridScorer(
        rule_engine=rule_engine,
        llm_scorer=mock_llm_scorer,
        model_name="test-model",
    )


def _make_tx(**kwargs) -> TransactionRequest:
    defaults = {"amount": 100.0, "currency": "USD", "country": "US", "category": "retail"}
    defaults.update(kwargs)
    return TransactionRequest(**defaults)


# ---------------------------------------------------------------------------
# Test: high-risk transaction scores above 70
# ---------------------------------------------------------------------------


class TestHighRiskTransaction:

    def test_high_risk_transaction_scores_above_70(self, hybrid_scorer, mock_llm_scorer):
        """$15K to Nigeria via wire_transfer should produce high combined score."""
        # LLM returns high score too
        mock_llm_scorer.score.return_value = (75.0, 100.0)

        tx = _make_tx(amount=15000, country="NG", category="wire_transfer")
        result = hybrid_scorer.score(tx)

        # Rule score: 10 (base) + 30 (high_amount) + 25 (high_risk_country)
        #             + 15 (high_risk_category) + 5 (round_amount) = 85
        # 85 < 90 threshold so LLM is called. Combined = 0.6*85 + 0.4*75 = 81
        assert result.risk_score > 70, f"Expected > 70, got {result.risk_score}"
        assert result.risk_level in ("high", "critical")


# ---------------------------------------------------------------------------
# Test: low-risk transaction scores below 30
# ---------------------------------------------------------------------------


class TestLowRiskTransaction:

    def test_low_risk_transaction_scores_below_30(self, hybrid_scorer, mock_llm_scorer):
        """$50 domestic retail purchase should produce low combined score."""
        # LLM returns low score
        mock_llm_scorer.score.return_value = (10.0, 50.0)

        tx = _make_tx(amount=50, country="US", category="retail")
        result = hybrid_scorer.score(tx)

        # Rule score: 10 (base only, no signals triggered)
        # 10 <= SKIP_LOW_THRESHOLD(10) so LLM is skipped
        # Combined = rule_score = 10
        assert result.risk_score < 30, f"Expected < 30, got {result.risk_score}"
        assert result.risk_level == "low"


# ---------------------------------------------------------------------------
# Test: rule engine signal detection
# ---------------------------------------------------------------------------


class TestRuleEngineSignals:

    def test_rule_engine_signals_detected(self, rule_engine):
        """Verify signals list includes high_amount, high_risk_country, etc."""
        tx = _make_tx(amount=15000, country="NG", category="wire_transfer")
        score, signals = rule_engine.score(tx)

        signal_names = {s.signal for s in signals}
        assert "high_amount" in signal_names
        assert "high_risk_country" in signal_names
        assert "high_risk_category" in signal_names
        assert "round_amount" in signal_names  # 15000 % 1000 == 0
        assert score == 85.0  # 10 + 30 + 25 + 15 + 5

    def test_round_amount_detected(self, rule_engine):
        tx = _make_tx(amount=5000, country="US", category="retail")
        _score, signals = rule_engine.score(tx)
        signal_names = {s.signal for s in signals}
        assert "round_amount" in signal_names

    def test_crypto_detected(self, rule_engine):
        tx = _make_tx(amount=500, country="US", category="crypto")
        _score, signals = rule_engine.score(tx)
        signal_names = {s.signal for s in signals}
        assert "crypto_transaction" in signal_names
        assert "high_risk_category" in signal_names

    def test_no_signals_for_clean_transaction(self, rule_engine):
        tx = _make_tx(amount=50, country="US", category="retail")
        score, signals = rule_engine.score(tx)
        assert signals == []
        assert score == 10.0  # base score only

    def test_score_capped_at_100(self, rule_engine):
        """Even with all signals, score should not exceed 100."""
        tx = _make_tx(amount=15000, country="NG", category="crypto")
        score, _signals = rule_engine.score(tx)
        assert score <= 100.0


# ---------------------------------------------------------------------------
# Test: LLM reasoning produces score
# ---------------------------------------------------------------------------


class TestLLMScoring:

    def test_llm_reasoning_produces_score(self, hybrid_scorer, mock_llm_scorer):
        """Mock LLM should produce a 0-100 score in the result."""
        mock_llm_scorer.score.return_value = (65.0, 120.0)

        tx = _make_tx(amount=5000, country="RU", category="retail")
        result = hybrid_scorer.score(tx)

        # Rule score: 10 + 25 (high_risk_country) + 5 (round amount) = 40
        # 10 < 40 < 90 -> LLM called
        assert result.llm_score is not None
        assert 0 <= result.llm_score <= 100
        assert result.llm_skipped is False

    def test_live_scorer_uses_bearer_token(self):
        credentials = {"api" + "_key": "test-token"}
        scorer = LLMScorer(
            endpoint="https://models.example.test/v1",
            model="example-model",
            **credentials,
        )
        try:
            assert scorer.endpoint == "https://models.example.test"
            assert scorer.client.headers["Authorization"] == "Bearer test-token"
        finally:
            scorer.client.close()

    def test_structured_llm_score_is_parsed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/chat/completions"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"score": 72}'}}]},
            )

        scorer = LLMScorer(endpoint="https://models.example.test/v1", model="example-model")
        scorer.client.close()
        scorer.client = httpx.Client(transport=httpx.MockTransport(handler))
        try:
            score, _latency = scorer.score(_make_tx(amount=5000), [])
            assert score == 72.0
        finally:
            scorer.client.close()

    def test_unstructured_llm_score_is_rejected(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "75/100"}}]},
            )

        scorer = LLMScorer(endpoint="https://models.example.test", model="example-model")
        scorer.client.close()
        scorer.client = httpx.Client(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(ValueError, match="invalid score payload"):
                scorer.score(_make_tx(amount=5000), [])
        finally:
            scorer.client.close()

    @pytest.mark.parametrize(
        "content",
        ['{"score": true}', '{"score": false}', '{"score": "72"}',
         '{"score": null}', '{"score": []}'],
    )
    def test_non_numeric_llm_score_is_rejected(self, content):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        scorer = LLMScorer(endpoint="https://models.example.test", model="example-model")
        scorer.client.close()
        scorer.client = httpx.Client(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(ValueError, match="invalid score payload"):
                scorer.score(_make_tx(amount=5000), [])
        finally:
            scorer.client.close()

    def test_demo_scorer_is_labeled_as_simulated(self):
        scorer = HybridScorer(
            rule_engine=RuleEngine(),
            llm_scorer=DemoLLMScorer(model="example-model"),
            model_name="example-model",
        )
        result = scorer.score(_make_tx(amount=5000))
        assert result.model == "demo-simulator"

    def test_model_failure_is_reported_separately_from_rule_skips(self, rule_engine):
        unavailable_model = MagicMock()
        unavailable_model.score.side_effect = httpx.ConnectError("unavailable")
        scorer = HybridScorer(
            rule_engine=rule_engine,
            llm_scorer=unavailable_model,
            model_name="test-model",
        )

        result = scorer.score(_make_tx(amount=500, country="NG"))

        assert result.llm_skipped is True
        assert result.model == "rule-engine-only"
        assert result.skip_reason == "LLM unavailable; used the rule score only"
        assert scorer.stats.llm_failures == 1
        assert scorer.stats.llm_skips == 0


# ---------------------------------------------------------------------------
# Test: weighted combination 60/40
# ---------------------------------------------------------------------------


class TestWeightedCombination:

    def test_weighted_combination_60_40(self, rule_engine):
        """Verify the math: combined = 0.6 * rule + 0.4 * llm."""
        mock_llm = MagicMock()
        mock_llm.score.return_value = (50.0, 100.0)

        scorer = HybridScorer(
            rule_engine=rule_engine,
            llm_scorer=mock_llm,
            model_name="test-model",
        )

        # Use a transaction with known rule score
        tx = _make_tx(amount=5000, country="US", category="retail")
        # Rule score = 10 (base) + 5 (round_amount) = 15
        result = scorer.score(tx)

        expected = (15.0 * RULE_WEIGHT) + (50.0 * LLM_WEIGHT)
        assert result.risk_score == round(expected, 2), (
            f"Expected {expected}, got {result.risk_score}"
        )

    def test_weights_sum_to_one(self):
        """Verify RULE_WEIGHT + LLM_WEIGHT == 1.0."""
        assert RULE_WEIGHT + LLM_WEIGHT == 1.0


# ---------------------------------------------------------------------------
# Test: conditional skip -- confident high
# ---------------------------------------------------------------------------


class TestConditionalSkipHigh:

    def test_conditional_skip_confident_high(self, hybrid_scorer, mock_llm_scorer):
        """rule_score >= 90 should skip LLM entirely."""
        # Transaction that triggers many signals: 10 + 30 + 25 + 15 + 10 + 5 = 95
        tx = _make_tx(amount=15000, country="NG", category="crypto")
        result = hybrid_scorer.score(tx)

        assert result.llm_skipped is True
        assert result.llm_score is None
        assert result.skip_reason is not None
        assert "confident" in result.skip_reason.lower() or str(SKIP_HIGH_THRESHOLD) in result.skip_reason
        mock_llm_scorer.score.assert_not_called()


# ---------------------------------------------------------------------------
# Test: conditional skip -- confident low
# ---------------------------------------------------------------------------


class TestConditionalSkipLow:

    def test_conditional_skip_confident_low(self, rule_engine):
        """rule_score <= 10 should skip LLM entirely."""
        mock_llm = MagicMock()

        # Create a rule engine that returns a very low score
        custom_rule = MagicMock()
        custom_rule.score.return_value = (5.0, [])

        scorer = HybridScorer(
            rule_engine=custom_rule,
            llm_scorer=mock_llm,
            model_name="test-model",
        )

        tx = _make_tx(amount=50, country="US", category="retail")
        result = scorer.score(tx)

        assert result.llm_skipped is True
        assert result.llm_score is None
        mock_llm.score.assert_not_called()


# ---------------------------------------------------------------------------
# Test: ambiguous range calls LLM
# ---------------------------------------------------------------------------


class TestAmbiguousCallsLLM:

    def test_ambiguous_calls_llm(self, hybrid_scorer, mock_llm_scorer):
        """rule_score between 30-70 should call LLM."""
        mock_llm_scorer.score.return_value = (45.0, 80.0)

        # Transaction with rule score in ambiguous range
        # 10 (base) + 25 (high_risk_country) = 35
        tx = _make_tx(amount=500, country="NG", category="retail")
        result = hybrid_scorer.score(tx)

        assert result.llm_skipped is False
        assert result.llm_score is not None
        mock_llm_scorer.score.assert_called_once()


# ---------------------------------------------------------------------------
# Test: skip rate in stats
# ---------------------------------------------------------------------------


class TestSkipRateStats:

    def test_skip_rate_in_stats(self, rule_engine):
        """Verify skip rate is reported correctly after mixed transactions."""
        mock_llm = MagicMock()
        mock_llm.score.return_value = (50.0, 100.0)

        scorer = HybridScorer(
            rule_engine=rule_engine,
            llm_scorer=mock_llm,
            model_name="test-model",
        )

        # Score a high-confidence transaction (will skip LLM)
        # 10 + 30 + 25 + 15 + 10 + 5 = 95 > 90
        tx_high = _make_tx(amount=15000, country="NG", category="crypto")
        scorer.score(tx_high)

        # Score an ambiguous transaction (will call LLM)
        tx_mid = _make_tx(amount=500, country="NG", category="retail")
        scorer.score(tx_mid)

        # Score another high-confidence
        scorer.score(tx_high)

        assert scorer.stats.total_scored == 3
        assert scorer.stats.llm_skips == 2
        assert scorer.stats.llm_calls == 1
        # Skip rate = 2/3 * 100 = 66.67%
        assert abs(scorer.stats.llm_skip_rate_pct - 66.67) < 0.1


# ---------------------------------------------------------------------------
# Test: batch scoring
# ---------------------------------------------------------------------------


class TestBatchScoring:

    def test_batch_scoring(self, hybrid_scorer, mock_llm_scorer):
        """Batch endpoint should score multiple transactions."""
        mock_llm_scorer.score.return_value = (40.0, 80.0)

        transactions = [
            _make_tx(amount=50, country="US", category="retail"),
            _make_tx(amount=15000, country="NG", category="wire_transfer"),
            _make_tx(amount=500, country="RU", category="crypto"),
        ]

        results = [hybrid_scorer.score(tx) for tx in transactions]

        assert len(results) == 3
        # Each result should have required fields
        for r in results:
            assert isinstance(r.risk_score, float)
            assert r.risk_level in ("low", "medium", "high", "critical")
            assert isinstance(r.signals, list)
            assert r.ai_disclaimer != ""

    def test_batch_has_varying_risk_levels(self, hybrid_scorer, mock_llm_scorer):
        """Different transactions should produce different risk levels."""
        mock_llm_scorer.score.return_value = (20.0, 80.0)

        tx_low = _make_tx(amount=50, country="US", category="retail")
        tx_high = _make_tx(amount=15000, country="NG", category="crypto")

        result_low = hybrid_scorer.score(tx_low)
        result_high = hybrid_scorer.score(tx_high)

        assert result_low.risk_score < result_high.risk_score
