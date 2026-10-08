# Forward model conventions (DISCO-DJ 2LPT + PNG + bias + RSD), $80^3$

res $=80$, $L=10000$, cell $=125$ Mpc/$h$, $z=1.651$, $b_1=2.44$, $p=1.6$, $b_\phi=2.832$, $b_2=0.830$, $b_K=-0.411$, $b_{\phi\delta}=1.867$
growth $D=0.4691$, $f=0.9416$; $k_{\rm Ny}=0.0251$
matter field (2LPT, no RSD): 1.9s first call; $P_m/P_{\rm lin}$ at $k<0.02$: 0.884
galaxy field $f_{\rm NL}=+0$: 1.1s; $P_g/[(b_1 + b_\phi f_{\rm NL}/T)^2 P_{\rm lin}]$ at $k<0.01$: 1.007, $0.01<k<0.02$: 0.774
galaxy field $f_{\rm NL}=+100$: 0.5s; $P_g/[(b_1 + b_\phi f_{\rm NL}/T)^2 P_{\rm lin}]$ at $k<0.01$: 1.026, $0.01<k<0.02$: 0.795
galaxy field $f_{\rm NL}=-100$: 0.5s; $P_g/[(b_1 + b_\phi f_{\rm NL}/T)^2 P_{\rm lin}]$ at $k<0.01$: 1.161, $0.01<k<0.02$: 0.748
RSD $\mu\in[0.00,0.33)$: $P_s(k,\mu)/P_{\rm real}(k,\mu)$ at $0.003<k<0.015$ = 1.029  (Kaiser $\langle(1+\beta\mu^2)^2\rangle$: 1.029)
RSD $\mu\in[0.33,0.66)$: $P_s(k,\mu)/P_{\rm real}(k,\mu)$ at $0.003<k<0.015$ = 1.199  (Kaiser $\langle(1+\beta\mu^2)^2\rangle$: 1.208)
RSD $\mu\in[0.66,1.00)$: $P_s(k,\mu)/P_{\rm real}(k,\mu)$ at $0.003<k<0.015$ = 1.555  (Kaiser $\langle(1+\beta\mu^2)^2\rangle$: 1.617)

Expectations: matter ratio $\approx1$ at $k < k_{\rm Ny}/3$ (CIC + 2LPT), galaxy ratio $\approx1$ at low $k$ for all $f_{\rm NL}$ (tests the PNG
normalisation $T_{\Phi\to\delta}$ and $b_\phi$), RSD ratios match Kaiser at low $k$ (observer placed far along $-z$ so the LOS is the $z$ axis).

Figure: `test/figures/forward_model_res80.png`
