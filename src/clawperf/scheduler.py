"""User arrival scheduler — burst / steady / poisson.

Yields (user_id, interval) where interval is the seconds to wait
BEFORE launching this user, relative to the previous user's launch.
"""

from __future__ import annotations

import random
from typing import AsyncIterator


async def burst_scheduler(num_users: int) -> AsyncIterator[tuple[int, float]]:
    """All users start immediately — interval is 0 for every user."""
    for uid in range(num_users):
        yield uid, 0.0


async def steady_scheduler(num_users: int, interval: float) -> AsyncIterator[tuple[int, float]]:
    """Users arrive every *interval* seconds. First user at t=0."""
    for uid in range(num_users):
        yield uid, interval if uid > 0 else 0.0


async def poisson_scheduler(num_users: int, lambda_rate: float) -> AsyncIterator[tuple[int, float]]:
    """Users arrive following a Poisson process. Intervals ~ Exp(lambda_rate)."""
    for uid in range(num_users):
        yield uid, random.expovariate(lambda_rate) if uid > 0 else 0.0


def get_scheduler(config):
    if config.arrival_mode == "burst":
        return burst_scheduler(config.num_users)
    elif config.arrival_mode == "steady":
        return steady_scheduler(config.num_users, config.arrival_param)
    elif config.arrival_mode == "poisson":
        return poisson_scheduler(config.num_users, config.arrival_param)
    raise ValueError(f"Unknown arrival mode: {config.arrival_mode}")