"""Synthetic processes used for the reproduction."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


@dataclass
class ProcessSample:
    name: str
    values: np.ndarray
    description: str


def brownian_motion(n: int, dt: float = 1.0, seed: int | None = None) -> ProcessSample:
    rng = np.random.default_rng(seed)
    increments = rng.normal(scale=math.sqrt(dt), size=n)
    values = np.cumsum(increments)
    return ProcessSample(
        name="brownian_motion",
        values=values,
        description="Brownian motion with unit variance increments",
    )


def signed_poisson_process(
    n: int,
    intensity: float = 0.5,
    dt: float = 1.0,
    seed: int | None = None,
) -> ProcessSample:
    rng = np.random.default_rng(seed)
    event_counts = rng.poisson(intensity * dt, size=n)
    signs = rng.choice([-1, 1], size=event_counts.sum())
    increments = np.zeros(n)
    idx = 0
    for t, count in enumerate(event_counts):
        if count == 0:
            continue
        increments[t] = signs[idx : idx + count].sum()
        idx += count
    values = np.cumsum(increments)
    return ProcessSample(
        name="signed_poisson",
        values=values,
        description=f"Signed Poisson process with intensity={intensity}",
    )


def _lognormal_volatility(n: int, rng: np.random.Generator, lambda2: float, decay: float) -> np.ndarray:
    white = rng.normal(size=n + 1024)
    kernel = 1.0 / np.sqrt(np.arange(1, n + 1025))
    kernel *= np.exp(-decay * np.arange(1, n + 1025))
    kernel /= np.linalg.norm(kernel)
    log_vol = np.convolve(white, kernel, mode="valid")[:n]
    log_vol -= log_vol.mean()
    log_vol *= math.sqrt(lambda2) / (log_vol.std() + 1e-12)
    return log_vol


def multifractal_random_walk(
    n: int,
    lambda2: float = 0.02,
    decay: float = 0.0005,
    dt: float = 1.0,
    seed: int | None = None,
    skew: float = 0.0,
    skew_decay: float = 0.01,
) -> ProcessSample:
    rng = np.random.default_rng(seed)
    log_vol = _lognormal_volatility(n, rng, lambda2=lambda2, decay=decay)
    increments = rng.normal(scale=math.sqrt(dt), size=n) * np.exp(log_vol)
    if skew != 0.0:
        k_len = min(n, 2048)
        kernel = np.exp(-skew_decay * np.arange(k_len))
        kernel[0] = 0.0
        asym = np.convolve(increments, kernel, mode="full")[:n]
        increments = increments + skew * asym
    values = np.cumsum(increments)
    name = "skewed_mrw" if skew != 0.0 else "mrw"
    description = "Multifractal random walk"
    if skew:
        description += f" with skew={skew}"
    return ProcessSample(name=name, values=values, description=description)


def quadratic_hawkes_process(
    n: int,
    baseline: float = 0.3,
    linear: float = 0.15,
    quadratic: float = 0.1,
    decay_linear: float = 0.05,
    decay_quadratic: float = 0.08,
    dt: float = 1.0,
    seed: int | None = None,
    max_intensity: float = 3.0,
    max_events: int = 5,
) -> ProcessSample:
    rng = np.random.default_rng(seed)
    lin_state = 0.0
    quad_state = 0.0
    increments = np.zeros(n)
    intensity = baseline
    for t in range(n):
        lam = max(intensity, 1e-6) * dt
        lam = min(lam, max_intensity)
        events = min(rng.poisson(lam), max_events)
        if events:
            increments[t] = rng.choice([-1, 1], size=events).sum()
        lin_state = math.exp(-decay_linear) * lin_state + abs(increments[t])
        quad_state = math.exp(-decay_quadratic) * quad_state + increments[t] ** 2
        intensity = baseline + linear * lin_state + quadratic * quad_state
        intensity = min(max(intensity, 0.01), max_intensity)
    values = np.cumsum(increments)
    return ProcessSample(
        name="quadratic_hawkes",
        values=values,
        description="Quadratic Hawkes process with exponential kernels",
    )


PROCESS_FACTORIES: Dict[str, Callable[[int, int], ProcessSample]] = {
    "brownian": lambda n, seed: brownian_motion(n, seed=seed),
    "poisson": lambda n, seed: signed_poisson_process(n, seed=seed),
    "mrw": lambda n, seed: multifractal_random_walk(n, seed=seed),
    "skewed_mrw": lambda n, seed: multifractal_random_walk(n, skew=0.2, seed=seed),
    "hawkes": lambda n, seed: quadratic_hawkes_process(n, seed=seed),
}
