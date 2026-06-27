"""Tests for the controlled hit-rate prompt builder."""

from __future__ import annotations

import random
from unittest.mock import MagicMock

import pytest

from clawperf.hitrate import (
    BOUNDARY_TOKENS,
    build_hitrate_requests,
    target_hit_rate,
)


def _mock_tm():
    """Tokenizer manager whose generate_random_content respects the global
    random seed (so distinct seeds -> distinct content), like the real one."""
    tm = MagicMock()
    tm.generate_random_content.side_effect = lambda n: "".join(
        random.choice("abcdefghij") for _ in range(max(1, n * 4))
    )
    return tm


def test_build_requests_count_and_shape():
    reqs = build_hitrate_requests(
        num_requests=10, input_len=100, prefix_len=40,
        prefix_num=2, tokenizer_manager=_mock_tm(), seed=1,
    )
    assert len(reqs) == 10
    for r in reqs:
        assert r.total_len == 100
        assert r.prefix_len == 40
        # prefill prompt is the shared prefix; measure prompt starts with it.
        assert r.measure_prompt.startswith(r.prefill_prompt)


def test_prefix_num_distinct_prefixes():
    """prefix_num distinct prefixes are generated; each reused round-robin."""
    reqs = build_hitrate_requests(
        num_requests=6, input_len=50, prefix_len=10,
        prefix_num=3, tokenizer_manager=_mock_tm(), seed=42,
    )
    distinct = {r.prefill_prompt for r in reqs}
    assert len(distinct) == 3
    # Each prefix used by exactly num_requests // prefix_num = 2 requests.
    from collections import Counter

    counts = Counter(r.prefill_prompt for r in reqs)
    assert all(c == 2 for c in counts.values())


def test_one_prefix_shared_by_all():
    """prefix_num=1 -> all requests share one prefix."""
    reqs = build_hitrate_requests(
        num_requests=5, input_len=50, prefix_len=10,
        prefix_num=1, tokenizer_manager=_mock_tm(), seed=7,
    )
    assert len({r.prefill_prompt for r in reqs}) == 1


def test_boundary_tokens_differ_per_request():
    """The 3-token boundary must be unique per request so the cache stops at prefix_len."""
    reqs = build_hitrate_requests(
        num_requests=5, input_len=50, prefix_len=10,
        prefix_num=1, tokenizer_manager=_mock_tm(), seed=3,
    )
    # The portion after the prefix (boundary + suffix) must differ across requests.
    tails = {r.measure_prompt[len(r.prefill_prompt):] for r in reqs}
    assert len(tails) == 5  # all unique


def test_validation_prefix_num_too_large():
    with pytest.raises(ValueError, match="prefix_num .* must be <= num_requests"):
        build_hitrate_requests(
            num_requests=5, input_len=50, prefix_len=10,
            prefix_num=10, tokenizer_manager=_mock_tm(),
        )


def test_validation_input_len_too_small():
    with pytest.raises(ValueError, match="input_len .* too small"):
        build_hitrate_requests(
            num_requests=5, input_len=13, prefix_len=10,  # 13 - 10 - 3 = 0
            prefix_num=1, tokenizer_manager=_mock_tm(),
        )


def test_target_hit_rate():
    assert target_hit_rate(50, 100) == 0.5
    assert target_hit_rate(0, 100) == 0.0
    assert target_hit_rate(50, 0) == 0.0


def test_boundary_tokens_constant():
    assert BOUNDARY_TOKENS == 3


def test_config_validate_hitrate_ok():
    from clawperf.config import BenchmarkConfig

    cfg = BenchmarkConfig(
        endpoint="http://x", model="m", mode="hitrate",
        num_requests=100, input_len=1024, output_len=128,
        hit_rate=0.5, prefix_num=10,
    )
    assert cfg.validate() == []


def test_config_validate_hitrate_errors():
    from clawperf.config import BenchmarkConfig

    cfg = BenchmarkConfig(
        endpoint="http://x", model="m", mode="hitrate",
        num_requests=5, input_len=10, prefix_num=10,  # prefix_num > num_requests
    )
    problems = cfg.validate()
    assert any("prefix_num" in p for p in problems)


def test_config_validate_unknown_mode():
    from clawperf.config import BenchmarkConfig

    cfg = BenchmarkConfig(endpoint="http://x", model="m", mode="bogus")
    assert any("unknown mode" in p for p in cfg.validate())
