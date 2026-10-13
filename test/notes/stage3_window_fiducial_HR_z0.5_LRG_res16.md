# Stage 3: grid-window consistency of NGP-gridded halos and the oversampled forward model

Sample fiducial_HR_z0.5_LRG, realisations 0-3, z = 0.5, measured $b_1$ = 1.880; 16$^3$ grid, cell 62.5 Mpc/h.
Data: CIC on the $64^3$ mesh truncated to the $16^3$ grid's Fourier modes, shot noise $L^3/N$ subtracted. Model: same downsampling, forward model at the data's
$b_1$ (Lazeyras $b_2$, $b_K = -2/7(b_1-1)$, $f_{\rm NL} = 0$), 16 random fields. Raw grid power, no window deconvolution, so model and
data should agree if the model's grid window matches the data's.

| k | 0.012 | 0.022 | 0.032 | 0.042 | 0.052 | 0.061 | 0.071 | 0.080 |
|---|---|---|---|---|---|---|---|---|
| data (r) | 3.94 | 3.62 | 3.27 | 3.29 | 2.68 | 1.69 | 0.47 | -0.56 |
| model os=4 (r) | 3.26 | 3.49 | 3.41 | 3.39 | 2.76 | 2.09 | 0.90 | 0.00 |
| data (s) | 5.04 | 4.73 | 4.25 | 4.26 | 3.48 | 2.21 | 0.70 | -0.56 |
| model os=4 (s) | 4.28 | 4.54 | 4.46 | 4.40 | 3.56 | 2.69 | 1.15 | 0.00 |

Figure: `test/figures/stage3_window_fiducial_HR_z0.5_LRG_res16.png`
