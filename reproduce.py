"""Reproduce empirical scattering spectra results from the paper."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.scattering.features import (
    compute_wavelet_statistics,
    fit_power_law,
    phase_modulus_cross_spectrum,
    scale_invariant_phase_modulus,
    scale_invariant_scattering,
    scattering_cross_spectrum,
)
from src.scattering.processes import PROCESS_FACTORIES, ProcessSample
from src.scattering.utils import complex_dict_to_json, save_json
from src.scattering.wavelets import build_wavelet_bank, wavelet_transform


def summarize_process(sample: ProcessSample, output_dir: Path) -> Dict:
    bank = build_wavelet_bank(len(sample.values), j_min=1, j_max=9)
    coeffs = wavelet_transform(sample.values, bank)
    stats = compute_wavelet_statistics(coeffs, bank.scales)
    slope_sigma, intercept_sigma = fit_power_law(stats.scales, stats.sigma2)
    slope_sparsity, intercept_sparsity = fit_power_law(stats.scales, stats.sparsity_squared)

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "process": sample.name,
        "description": sample.description,
        "sigma2_slope": float(slope_sigma),
        "sigma2_intercept": float(intercept_sigma),
        "sparsity2_slope": float(slope_sparsity),
        "sparsity2_intercept": float(intercept_sparsity),
        "average_sparsity": float(np.mean(stats.sparsity)),
    }

    np.savez(
        output_dir / f"{sample.name}_wavelet_stats.npz",
        scales=np.asarray(stats.scales),
        sigma2=stats.sigma2,
        sparsity2=stats.sparsity_squared,
        sparsity=stats.sparsity,
    )

    cross = phase_modulus_cross_spectrum(coeffs, stats, max_scale_shift=4)
    invariant_cross = scale_invariant_phase_modulus(cross)
    save_json({str(k): [complex(val).real, complex(val).imag] for k, val in invariant_cross.items()}, output_dir / f"{sample.name}_phase_modulus.json")

    scatter = scattering_cross_spectrum(coeffs, bank, stats, max_scale_shift=3, max_log_frequency_shift=5)
    invariant_scatter = scale_invariant_scattering(scatter)
    save_json(complex_dict_to_json(invariant_scatter), output_dir / f"{sample.name}_scattering.json")

    summary["phase_modulus"] = {str(k): complex(val).real for k, val in invariant_cross.items()}
    summary["phase_modulus_imag"] = {str(k): complex(val).imag for k, val in invariant_cross.items()}
    summary["scattering"] = complex_dict_to_json(invariant_scatter)

    return summary


def plot_wavelet_statistics(process_summaries: List[Dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for summary in process_summaries:
        stats_path = output_dir / f"{summary['process']}_wavelet_stats.npz"
        data = np.load(stats_path)
        scales = data["scales"]
        axes[0].plot(scales, np.log2(data["sigma2"]), label=summary["process"])
        axes[1].plot(scales, np.log2(data["sparsity2"]), label=summary["process"])
    axes[0].set_title("Wavelet power spectrum log2")
    axes[0].set_xlabel("Scale j")
    axes[0].set_ylabel("log2 sigma^2")
    axes[1].set_title("Wavelet sparsity factor log2")
    axes[1].set_xlabel("Scale j")
    axes[1].set_ylabel("log2 s^2")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "wavelet_statistics.png", dpi=200)
    plt.close(fig)


def main() -> None:
    np.random.seed(1234)
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    summaries: List[Dict] = []
    for name, factory in PROCESS_FACTORIES.items():
        sample = factory(16384, seed=42)
        summary = summarize_process(sample, output_dir)
        save_json(summary, output_dir / f"{sample.name}_summary.json")
        summaries.append(summary)

    plot_wavelet_statistics(summaries, output_dir)

    table_path = output_dir / "summary_table.json"
    save_json(summaries, table_path)
    print(f"Saved summaries to {table_path}")


if __name__ == "__main__":
    main()
