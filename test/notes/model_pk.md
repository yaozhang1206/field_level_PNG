# P(k) model sanity check (Chaussidon+24 Eq. 2.4)

Model: $P(k,\mu) = \dfrac{[b_1 + b_\phi f_{\rm NL}/T(k) + f\mu^2]^2}{[1 + (k\mu\Sigma_s)^2/2]^2}\,P_{\rm lin} + s_{n0}$, $\Sigma_s = 4$ Mpc/$h$.
Kaiser quadrupole ratio for $f_{\rm NL}=0$ should be $\approx \dfrac{\tfrac43\beta + \tfrac47\beta^2}{1 + \tfrac23\beta + \tfrac15\beta^2}$ at low $k$.

- QSO FKP: $z=1.651$, $b_1=2.44$, $p=1.6$: $f=0.942$, $b_\phi=2.832$, $\Delta b(k=0.005)=0.897$ for $f_{\rm NL}=50$, $P_0(k=0.02)=40178$
- LRG FKP: $z=0.733$, $b_1=2.06$, $p=1.0$: $f=0.826$, $b_\phi=3.574$, $\Delta b(k=0.005)=0.775$ for $f_{\rm NL}=50$, $P_0(k=0.02)=61746$
- QSO: expected low-$k$ $P_2/P_0 = 0.466$
- LRG: expected low-$k$ $P_2/P_0 = 0.482$

Figure: `test/figures/model_pk.png`
