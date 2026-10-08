"""Shared Stage-1 Part B configuration: redshift shells, bias templates, geometry."""
import numpy as np
from cosmology import Cosmology
from geometry import Geometry, dr1_angular_mask
from forward_model import default_shell

B1_OF_Z = {"LRG": lambda z: 0.209 * (1 + z) ** 2 + 1.415, "QSO": lambda z: 0.237 * (1 + z) ** 2 + 0.771}
P_UMF = {"LRG": 1.0, "QSO": 1.6}
SHELL_EDGES = {"QSO": [(0.8, 1.3), (1.3, 2.1), (2.1, 3.1)], "LRG": [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)]}
P0 = {"LRG": 5e4, "QSO": 3e4}


def build(cosmo=None, tracers=("QSO", "LRG"), p_qso=None):
    cosmo = cosmo or Cosmology(backend="class")
    geoms = {t: Geometry(cosmo, t, min(e[0] for e in SHELL_EDGES[t]), max(e[1] for e in SHELL_EDGES[t]), nbins=60)
             for t in tracers}
    shells = []
    for t in tracers:
        p = P_UMF[t] if (p_qso is None or t != "QSO") else p_qso
        for zlo, zhi in SHELL_EDGES[t]:
            zc = 0.5 * (zlo + zhi)
            shells.append(default_shell(t, zlo, zhi, float(B1_OF_Z[t](zc)), p))
    nz_funcs = {t: (lambda zz, g=geoms[t]: np.interp(zz, g.z, g.nz_obs(), left=0.0, right=0.0)) for t in tracers}
    return cosmo, geoms, shells, nz_funcs
