# P(k) model sanity check (Chaussidon+24 Eq. 2.4)

Model: P(k,mu) = [b1 + b_phi f_NL / T(k) + f mu^2]^2 / [1 + (k mu Sigma_s)^2/2]^2 P_lin + s_n0, Sigma_s=4 Mpc/h.
Kaiser quadrupole ratio for f_NL=0 should be ~ (4/3 beta + 4/7 beta^2)/(1 + 2/3 beta + beta^2/5) at low k.

- QSO FKP: z=1.651, b1=2.44, p=1.6: f=0.942, b_phi=2.832, Delta b(k=0.005)=0.897 for f_NL=50, P0(k=0.02)=40178
- LRG FKP: z=0.733, b1=2.06, p=1.0: f=0.826, b_phi=3.574, Delta b(k=0.005)=0.775 for f_NL=50, P0(k=0.02)=61746
- QSO: expected low-k P2/P0 = 0.466
- LRG: expected low-k P2/P0 = 0.482

Figure: `test/figures/model_pk.png`
