# Stage 3: grid-window consistency of NGP-gridded halos and the oversampled forward model

Sample fiducial_HR_z0.5_LRG, realisations 0-3, z = 0.5, measured $b_1$ = 1.880; 16$^3$ grid, cell 62.5 Mpc/h.
Data: CIC on the $64^3$ mesh truncated to the $16^3$ grid's Fourier modes, shot noise $L^3/N$ subtracted. Model: same downsampling, forward model at the data's
$b_1$ (Lazeyras $b_2$, $b_K = -2/7(b_1-1)$, $f_{\rm NL} = 0$), 16 random fields. Raw grid power, no window deconvolution, so model and
data should agree if the model's grid window matches the data's.

| k | 0.012 | 0.022 | 0.032 | 0.042 | 0.052 | 0.061 | 0.071 | 0.080 |
|---|---|---|---|---|---|---|---|---|
| data (r) | 3.94 | 3.62 | 3.27 | 3.29 | 2.68 | 1.69 | 0.47 | -0.56 |
| model p1m2 (r) | 3.21 | 3.32 | 3.10 | 2.90 | 2.23 | 1.60 | 0.65 | 0.00 |
| model p1m4 (r) | 3.27 | 3.49 | 3.41 | 3.40 | 2.77 | 2.10 | 0.91 | 0.00 |
| model p2m2 (r) | 3.21 | 3.32 | 3.09 | 2.88 | 2.21 | 1.58 | 0.64 | 0.00 |
| model p2m4 (r) | 3.26 | 3.49 | 3.41 | 3.39 | 2.76 | 2.09 | 0.90 | 0.00 |
| model p4m4 (r) | 3.26 | 3.49 | 3.41 | 3.39 | 2.76 | 2.09 | 0.90 | 0.00 |
| data (s) | 5.04 | 4.73 | 4.25 | 4.26 | 3.48 | 2.21 | 0.70 | -0.56 |
| model p1m2 (s) | 4.20 | 4.30 | 4.00 | 3.70 | 2.81 | 1.99 | 0.80 | 0.00 |
| model p1m4 (s) | 4.29 | 4.55 | 4.46 | 4.41 | 3.57 | 2.70 | 1.16 | 0.00 |
| model p2m2 (s) | 4.20 | 4.30 | 3.99 | 3.66 | 2.77 | 1.95 | 0.77 | 0.00 |
| model p2m4 (s) | 4.28 | 4.54 | 4.46 | 4.40 | 3.56 | 2.69 | 1.14 | 0.00 |
| model p4m4 (s) | 4.28 | 4.54 | 4.46 | 4.40 | 3.56 | 2.69 | 1.15 | 0.00 |

Figure: `test/figures/stage3_window_fiducial_HR_z0.5_LRG_res16.png`
