# Geometry: n(z), area, V_eff for DR1 LRG/QSO

## LRG  (0.4 < z < 1.1)
- N (weighted) = 2,144,020  (paper: 2,130,621); pixel area = 10102 deg^2, effective area (N/int NX dV) = 5417 deg^2 -> 396 deg^-2 (paper 530)
- V_total(eff area) = 7.44 (Gpc/h)^3; V_eff(P0=5e+04) = 6.20 [NX, eff area] / 10.15 [count, pixel area] (Gpc/h)^3
- nbar_eff = 3.04e-04 [NX] / 1.68e-04 [count]; z_eff(FKP-weighted) = 0.792

## QSO  (0.8 < z < 3.1)
- N (weighted) = 1,181,972  (paper: 1,189,129); pixel area = 10078 deg^2, effective area (N/int NX dV) = 7159 deg^2 -> 165 deg^-2 (paper 185)
- V_total(eff area) = 58.70 (Gpc/h)^3; V_eff(P0=3e+04) = 8.29 [NX, eff area] / 7.64 [count, pixel area] (Gpc/h)^3
- nbar_eff = 2.48e-05 [NX] / 1.78e-05 [count]; z_eff(FKP-weighted) = 1.718

Notes: NX is the catalogue density used for FKP weights (completeness-scaled). The 'count' estimate uses
the HEALPix (nside=128) footprint of the galaxy catalogue itself; DR1 tiling is incomplete so this pixel area
overestimates the effective area and the count-based n(z) is too low. The NX-based V_eff with the effective area
is the one used in the Fisher. z_eff here is V_eff-weighted, not the pair-weighted z_eff of the paper.

Figure: `test/figures/geometry_nz.png`
