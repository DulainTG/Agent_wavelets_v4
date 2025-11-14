"""Scattering-based summary statistics used in the reproduction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .wavelets import WaveletBank, scattering_transform


@dataclass
class WaveletStatistics:
    scales: np.ndarray
    sigma2: np.ndarray
    sigma: np.ndarray
    mean_modulus: np.ndarray
    sparsity: np.ndarray
    sparsity_squared: np.ndarray


def compute_wavelet_statistics(coeffs: np.ndarray, scales: np.ndarray) -> WaveletStatistics:
    if coeffs.ndim != 2:
        raise ValueError("Wavelet coefficients must be a 2-D array")
    modulus = np.abs(coeffs)
    sigma2 = modulus**2.0
    sigma2 = sigma2.mean(axis=1)
    sigma = np.sqrt(sigma2)
    mean_modulus = modulus.mean(axis=1)
    sparsity = mean_modulus / (sigma + 1e-12)
    sparsity_squared = (mean_modulus**2) / (sigma2 + 1e-12)
    return WaveletStatistics(
        scales=np.asarray(scales, dtype=float),
        sigma2=sigma2,
        sigma=sigma,
        mean_modulus=mean_modulus,
        sparsity=sparsity,
        sparsity_squared=sparsity_squared,
    )


def fit_power_law(scales: np.ndarray, values: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(scales, dtype=float)
    y = np.log2(np.asarray(values))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept


def phase_modulus_cross_spectrum(
    coeffs: np.ndarray, stats: WaveletStatistics, max_scale_shift: int
) -> Dict[int, np.ndarray]:
    scales = np.asarray(stats.scales, dtype=int)
    sigma = stats.sigma
    num_scales = coeffs.shape[0]
    results: Dict[int, np.ndarray] = {}
    for a in range(max_scale_shift + 1):
        if a >= num_scales:
            break
        values = []
        for j in range(num_scales - a):
            numerator = np.mean(coeffs[j] * np.abs(coeffs[j + a]))
            denom = sigma[j] * sigma[j + a] + 1e-12
            values.append(numerator / denom)
        results[a] = np.asarray(values)
    return results


def scale_invariant_phase_modulus(cross_spectrum: Dict[int, np.ndarray]) -> Dict[int, complex]:
    return {a: values.mean() for a, values in cross_spectrum.items()}


def scattering_cross_spectrum(
    coeffs: np.ndarray,
    bank: WaveletBank,
    stats: WaveletStatistics,
    max_scale_shift: int,
    max_log_frequency_shift: int,
) -> Dict[Tuple[int, int], complex]:
    num_scales = coeffs.shape[0]
    scales = np.asarray(bank.scales)
    if len(scales) != num_scales:
        raise ValueError("Scale list and coefficient count mismatch")
    modulus = np.abs(coeffs)
    scatter_cache = []
    for j in range(num_scales):
        min_j = scales[j] + 1
        scatter_cache.append(scattering_transform(modulus[j], bank, min_j=min_j))
    sigma = stats.sigma
    results: Dict[Tuple[int, int], complex] = {}
    for a in range(max_scale_shift + 1):
        for b in range(1, max_log_frequency_shift + 1):
            values = []
            for j in range(num_scales - a - b):
                idx_a = j + a
                idx_b = j + b
                idx_ab = j + a + b
                Sj = scatter_cache[j]
                Sja = scatter_cache[idx_a]
                row1 = Sj[idx_b]
                row2 = Sja[idx_ab]
                numerator = np.mean(row1 * np.conj(row2))
                denom = sigma[j] * sigma[idx_a] + 1e-12
                values.append(numerator / denom)
            if values:
                results[(a, b)] = np.mean(values)
    return results


def scale_invariant_scattering(scattering: Dict[Tuple[int, int], complex]) -> Dict[int, Dict[int, complex]]:
    aggregated: Dict[int, Dict[int, complex]] = {}
    for (a, b), value in scattering.items():
        aggregated.setdefault(a, {})[b] = value
    return aggregated
