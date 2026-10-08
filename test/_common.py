"""Shared helpers for tests: paths, figure/note writers."""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIG = ROOT / "test" / "figures"; NOTES = ROOT / "test" / "notes"
FIG.mkdir(exist_ok=True, parents=True); NOTES.mkdir(exist_ok=True, parents=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})

def save_fig(fig, name):
    p = FIG / f"{name}.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p

def write_note(name, title, lines):
    p = NOTES / f"{name}.md"
    with open(p, "w") as f:
        f.write(f"# {title}\n\n")
        for l in lines: f.write(l + "\n")
        f.write(f"\nFigure: `test/figures/{name}.png`\n")
    return p
