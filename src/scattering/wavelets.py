"""Wavelet filter bank and transforms for scattering reproduction."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class WaveletBank:
    """Collection of complex wavelet filters in the Fourier domain."""

    fft_length: int
    scales: List[int]
    fourier_filters: np.ndarray  # shape (num_scales, fft_length)
    fft_frequencies: np.ndarray

    @property
    def num_scales(self) -> int:
        return len(self.scales)


def _morlet_fourier(freqs: np.ndarray, scale: float, central_freq: float = 6.0, bandwidth: float = 1.0) -> np.ndarray:
    """Return the Fourier response of an analytic Morlet wavelet."""
    scaled = scale * freqs
    envelope = np.exp(-0.5 * ((scaled - central_freq) / (bandwidth * central_freq)) ** 2)
    positive = (scaled >= 0).astype(float)
    psi_hat = envelope * positive
    energy = np.sqrt(np.sum(np.abs(psi_hat) ** 2) / psi_hat.size)
    if energy == 0:
        return psi_hat
    return psi_hat / energy


def build_wavelet_bank(
    n: int,
    j_min: int = 0,
    j_max: int | None = None,
    central_freq: float = 6.0,
    bandwidth: float = 1.0,
) -> WaveletBank:
    """Construct a dyadic wavelet bank using analytic Morlet filters."""
    if j_max is None:
        j_max = int(math.log2(n)) - 2
        j_max = max(j_max, j_min)
    freqs = 2 * math.pi * np.fft.fftfreq(n)
    filters = []
    scales = []
    for j in range(j_min, j_max + 1):
        scale = 2.0 ** (-j)
        filters.append(_morlet_fourier(freqs, scale, central_freq=central_freq, bandwidth=bandwidth))
        scales.append(j)
    return WaveletBank(fft_length=n, scales=scales, fourier_filters=np.vstack(filters), fft_frequencies=freqs)


def wavelet_transform(x: np.ndarray, bank: WaveletBank) -> np.ndarray:
    """Compute complex wavelet transform at all dyadic scales."""
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if x.size != bank.fft_length:
        raise ValueError("Signal length and bank length mismatch")
    x_hat = np.fft.fft(x)
    coeffs = []
    for psi_hat in bank.fourier_filters:
        w_hat = x_hat * psi_hat
        coeffs.append(np.fft.ifft(w_hat))
    return np.vstack(coeffs)


def scattering_transform(modulus: np.ndarray, bank: WaveletBank, min_j: int = 0) -> np.ndarray:
    """Apply a second wavelet transform to modulus coefficients."""
    if modulus.ndim != 1:
        raise ValueError("Modulus input must be one-dimensional")
    if modulus.size != bank.fft_length:
        raise ValueError("Modulus length and bank length mismatch")
    mod_hat = np.fft.fft(modulus)
    coeffs = []
    for j, psi_hat in zip(bank.scales, bank.fourier_filters):
        if j < min_j:
            coeffs.append(np.zeros_like(modulus, dtype=complex))
            continue
        w_hat = mod_hat * psi_hat
        coeffs.append(np.fft.ifft(w_hat))
    return np.vstack(coeffs)
