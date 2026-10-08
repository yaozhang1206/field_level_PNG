# Dense 16^3 check: Schur (CG vs dense) vs exact marginal vs mode sums

- LRG (n=3.0e-04): sigma(f_NL) [b1 marg.]: CG-Schur 0.777 | dense Schur 0.777 | dense exact marginal 0.614 | Hutchinson(16 probes) 0.617 | mode sum exact 0.614 | mode sum Laplace-weighted 0.729
    F_ff: CG 2.3281 dense 2.3281 | marginal 3.6963 Hutchinson 3.6576 exact-sum 3.6963 | Laplace-sum 2.5231
- QSO (n=2.5e-05): sigma(f_NL) [b1 marg.]: CG-Schur 14.932 | dense Schur 14.932 | dense exact marginal 26.728 | Hutchinson(16 probes) 26.655 | mode sum exact 26.728 | mode sum Laplace-weighted 14.553
    F_ff: CG 0.0071 dense 0.0071 | marginal 0.0027 Hutchinson 0.0027 exact-sum 0.0027 | Laplace-sum 0.0072

Conclusions: CG operator code == dense Schur; dense exact marginal == Tier 1 mode sum (validates both);
Schur == Laplace-weighted sum (within one-realisation scatter): the Schur/profile estimate is biased by (P+N)/(2P) per mode;
Hutchinson reproduces the exact marginal within probe noise.

No figure (numerical table).

Figure: `test/figures/schur_dense16.png`
