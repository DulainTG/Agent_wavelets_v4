# Key definitions extracted from *Scale Dependencies and Self-Similar Models with Wavelet Scattering Spectra*

1. **Wavelet spectrum and sparsity.** The wavelet power spectrum is defined as
   $\sigma_W^2(j) = \mathbb{E}\{|X * \psi_j|^2\}$ while the sparsity factor is
   $s_W(j) = \mathbb{E}\{|X * \psi_j|\} / \sigma_W(j)$. Their square $s_W^2(j)$
   follows a power-law for wide-sense self-similar processes (Eq. 18).

2. **Scale-invariant phase-modulus cross-spectrum.** The normalized
   correlations $C_{W_j W_j}(a) = \mathbb{E}\{X * \psi_j \; |X * \psi_{j+a}|\}/\sigma_W(j)\sigma_W(j+a)$
   are averaged across scales to define a scale-invariant descriptor (Eq. 19).

3. **Scattering cross-spectrum.** Applying a second wavelet transform to
   $|X * \psi_j|$ yields coefficients whose diagonal correlations define
   $C_S(j; a, b)$. Averaging across scales gives the scale-invariant scattering
   cross-spectrum $C_S(a, b)$ (Eqs. 21–22).
