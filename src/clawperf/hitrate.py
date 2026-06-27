"""Controlled prefix-cache hit-rate test mode.

Builds prompts with a known shared-prefix / unique-suffix split so the target
hit rate is realized *by construction*, then measures the *actual* hit rate
from the server's Prometheus counters (delta of vllm:prefix_cache_*_total).

Prompt shape (per request, aisbench-style):
    [shared prefix] [3 boundary tokens] [unique suffix]
                       ^- unique per request, forces the cache to stop at
                          exactly prefix_len (no accidental continuation)

Distinct prefixes are assigned round-robin (request i -> prefixes[i % prefix_num]),
then the request list is shuffled so prefix groups are interleaved (vLLM-style),
exercising real cache reuse under concurrency rather than back-to-back duplicates.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

BOUNDARY_TOKENS = 3  # unique per-request tokens that terminate the cached prefix


@dataclass
class HitRateRequest:
    """One request in a hit-rate test.

    ``prefill_prompt`` is the prefix-only text sent in the prefill phase
    (injects the prefix into the KV cache). ``measure_prompt`` is the full
    prompt sent in the measure phase. Both share the same leading prefix text
    so the measure request hits on ``prefix_len`` tokens.
    """

    index: int
    prefix_idx: int
    prefix_len: int
    total_len: int
    prefill_prompt: str
    measure_prompt: str


def _random_token_text(tokenizer_manager, n_tokens: int, seed: int) -> str:
    """Deterministic random token content of ~n_tokens tokens (seeded)."""
    import random as _r

    saved_state = _r.getstate()
    try:
        _r.seed(seed)
        return tokenizer_manager.generate_random_content(n_tokens)
    finally:
        _r.setstate(saved_state)


def build_hitrate_requests(
    *,
    num_requests: int,
    input_len: int,
    prefix_len: int,
    prefix_num: int,
    tokenizer_manager,
    seed: int = 0,
) -> List[HitRateRequest]:
    """Construct the request list for a hit-rate test.

    Args:
        num_requests: total measure-phase requests.
        input_len: total prompt length (prefix + boundary + suffix).
        prefix_len: shared-prefix length (the part intended to cache-hit).
        prefix_num: number of DISTINCT prefixes; requests-per-prefix =
            num_requests // prefix_num. 1 = all share one prefix.
        tokenizer_manager: provides generate_random_content (random token ids).
        seed: reproducibility seed.

    Each request = distinct_prefix[i % prefix_num] + BOUNDARY unique tokens +
    unique suffix of (input_len - prefix_len - BOUNDARY) tokens.
    """
    if prefix_num < 1:
        raise ValueError("prefix_num must be >= 1")
    if prefix_num > num_requests:
        raise ValueError(
            f"prefix_num ({prefix_num}) must be <= num_requests ({num_requests})"
        )
    suffix_len = input_len - prefix_len - BOUNDARY_TOKENS
    if suffix_len < 1:
        raise ValueError(
            f"input_len ({input_len}) too small: need > prefix_len "
            f"({prefix_len}) + boundary ({BOUNDARY_TOKENS})"
        )

    # Distinct prefixes (cached once). Each is ~prefix_len random tokens.
    prefixes = [
        _random_token_text(tokenizer_manager, prefix_len, seed=seed + 1000 + i)
        for i in range(prefix_num)
    ]

    requests: List[HitRateRequest] = []
    for i in range(num_requests):
        pidx = i % prefix_num
        # Per-request unique boundary + suffix (seed-derived so reproducible).
        boundary = _random_token_text(
            tokenizer_manager, BOUNDARY_TOKENS, seed=seed + 2000 + i
        )
        suffix = _random_token_text(
            tokenizer_manager, suffix_len, seed=seed + 3000 + i
        )
        measure_prompt = prefixes[pidx] + boundary + suffix
        requests.append(
            HitRateRequest(
                index=i,
                prefix_idx=pidx,
                prefix_len=prefix_len,
                total_len=input_len,
                prefill_prompt=prefixes[pidx],
                measure_prompt=measure_prompt,
            )
        )

    # Interleave prefix groups so reuse happens under concurrency, not as
    # back-to-back duplicates (matches vLLM prefix_repetition's shuffle).
    rng = random.Random(seed)
    rng.shuffle(requests)
    return requests


def target_hit_rate(prefix_len: int, input_len: int) -> float:
    """The hit rate this construction targets (= prefix_len / input_len)."""
    if input_len <= 0:
        return 0.0
    return prefix_len / input_len
