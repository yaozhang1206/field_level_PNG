# Dense $16^3$ check: Schur (CG vs dense) vs exact marginal vs mode sums

Dense Schur: $F_{\rm marg} = F_{\theta\theta} - F_{\theta\varepsilon}F_{\varepsilon\varepsilon}^{-1}F_{\varepsilon\theta}$ with $F = J^TN^{-1}J + I$;
exact marginal: $F_{ij} = \tfrac12{\rm Tr}[C^{-1}\partial_iC\,C^{-1}\partial_jC]$, $C = J_\varepsilon J_\varepsilon^T + N$;
mode sums per independent mode: exact $\partial_iP\,\partial_jP/(P+N)^2$, Laplace-weighted $\partial_iP\,\partial_jP/(2P(P+N))$.

- LRG ($\bar n=3.0e-04$): $\sigma(f_{\rm NL})$ [$b_1$ marg.]: CG-Schur 0.777 | dense Schur 0.777 | dense exact marginal 0.614 | Hutchinson(16 probes) 0.617 | mode sum exact 0.614 | mode sum Laplace-weighted 0.729
    $F_{ff}$: CG 2.3281 dense 2.3281 | marginal 3.6963 Hutchinson 3.6576 exact-sum 3.6963 | Laplace-sum 2.5231
- QSO ($\bar n=2.5e-05$): $\sigma(f_{\rm NL})$ [$b_1$ marg.]: CG-Schur 14.932 | dense Schur 14.932 | dense exact marginal 26.728 | Hutchinson(16 probes) 26.655 | mode sum exact 26.728 | mode sum Laplace-weighted 14.553
    $F_{ff}$: CG 0.0071 dense 0.0071 | marginal 0.0027 Hutchinson 0.0027 exact-sum 0.0027 | Laplace-sum 0.0072

Conclusions: CG operator code $=$ dense Schur; dense exact marginal $=$ Tier 1 mode sum (validates both);
Schur $=$ Laplace-weighted sum (within one-realisation scatter): the Schur/profile estimate is biased by $(P+N)/(2P)$ per mode;
Hutchinson reproduces the exact marginal within probe noise.

No figure (numerical table).

Figure: `test/figures/schur_dense16.png`
