# Stage 3: grid-window consistency of NGP-gridded halos and the oversampled forward model

Sample fiducial_HR_z0.5_LRG, realisations 0-3, z = 0.5, measured $b_1$ = 1.880; 16$^3$ grid, cell 62.5 Mpc/h.
Data: NGP counts (cells aligned with the block-averaged fine lattice), shot noise $L^3/N$ subtracted. Model: forward model at the data's
$b_1$ (Lazeyras $b_2$, $b_K = -2/7(b_1-1)$, $f_{\rm NL} = 0$), 16 random fields. Raw grid power, no window deconvolution, so model and
data should agree if the model's grid window matches the data's.

| k | 0.012 | 0.022 | 0.032 | 0.042 | 0.052 | 0.061 | 0.071 | 0.080 |
|---|---|---|---|---|---|---|---|---|
| data (r) | 3.77 | 3.33 | 2.69 | 2.63 | 2.43 | 2.16 | 2.00 | 2.23 |
| model os=1 (r) | 3.03 | 2.81 | 2.25 | 1.73 | 1.30 | 1.05 | 0.88 | 0.78 |
| model os=2 (r) | 3.07 | 2.94 | 2.42 | 1.88 | 1.18 | 0.67 | 0.27 | 0.09 |
| model os=4 (r) | 3.09 | 3.00 | 2.50 | 1.96 | 1.22 | 0.68 | 0.25 | 0.07 |
| data (s) | 4.82 | 4.32 | 3.49 | 3.40 | 3.11 | 2.79 | 2.61 | 2.99 |
| model os=1 (s) | 3.93 | 3.56 | 2.78 | 2.07 | 1.51 | 1.18 | 0.96 | 0.83 |
| model os=2 (s) | 4.03 | 3.81 | 3.13 | 2.41 | 1.52 | 0.87 | 0.37 | 0.16 |
| model os=4 (s) | 4.06 | 3.90 | 3.28 | 2.56 | 1.60 | 0.90 | 0.36 | 0.13 |

Figure: `test/figures/stage3_window_fiducial_HR_z0.5_LRG_res16.png`
