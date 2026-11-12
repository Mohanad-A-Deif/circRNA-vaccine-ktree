"""Schematic of the biological mechanism and its abstraction in the decision model."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon, FancyBboxPatch, FancyArrowPatch, Rectangle

NAVY, CRIMSON, BLACK, FOREST, GREY = "#1f3a93", "#a31515", "#111111", "#1e6b3a", "#6b6b6b"
PALE_B, PALE_G, PALE_R, PALE_K = "#e9eef8", "#e8f2eb", "#f7e8e8", "#f2f2f2"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "mathtext.fontset": "dejavuserif", "font.size": 7.2,
                     "savefig.dpi": 600, "savefig.bbox": "tight"})

fig = plt.figure(figsize=(7.2, 5.3))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 144); ax.set_ylim(0, 106); ax.set_aspect("equal"); ax.axis("off")


def arrow(p, q, col=BLACK, ls="-", lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=col, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def txt(x, y, s, **k):
    k.setdefault("ha", "center"); k.setdefault("va", "center")
    ax.text(x, y, s, **k)


def box(x0, y0, w, h, fc, ec):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=fc, ec=ec, lw=0.9))


# ============================== panel (a) ==============================
txt(1, 103.5, "(a) Biological mechanism of a therapeutic circRNA cancer vaccine",
    ha="left", fontweight="bold", fontsize=8)

# --- LNP with circular RNA
cx, cy, r = 11, 80, 6.5
ax.add_patch(Circle((cx, cy), r, fc=PALE_K, ec="none"))
for th in np.linspace(0, 2 * np.pi, 30, endpoint=False):
    hx, hy = cx + r * np.cos(th), cy + r * np.sin(th)
    ax.add_patch(Circle((hx, hy), 0.75, fc=GREY, ec=BLACK, lw=0.3))
    ax.plot([hx - 1.6 * np.cos(th), hx - 0.7 * np.cos(th)], [hy - 1.6 * np.sin(th), hy - 0.7 * np.sin(th)],
            color=GREY, lw=0.5)
ax.add_patch(Circle((cx, cy), 2.6, fc="none", ec=CRIMSON, lw=1.8))
for th in np.linspace(0, 2 * np.pi, 12, endpoint=False):
    ax.plot([cx + 2.6 * np.cos(th), cx + 3.3 * np.cos(th)], [cy + 2.6 * np.sin(th), cy + 3.3 * np.sin(th)],
            color=CRIMSON, lw=0.6)
txt(cx, 91, "LNP–circRNA dose", fontweight="bold")
txt(cx, 69.5, "dose $d_j$ at $\\tau_j$\ntemplate $R$", color=CRIMSON)

# --- antigen-presenting cell
dx, dy = 37, 80
th = np.linspace(0, 2 * np.pi, 400)
rr = 6.2 + 1.9 * np.cos(6 * th) ** 3 + 0.4 * np.sin(11 * th)
ax.add_patch(Polygon(np.c_[dx + rr * np.cos(th), dy + rr * np.sin(th)], fc=PALE_B, ec=NAVY, lw=0.9))
ax.add_patch(Ellipse((dx + 1.0, dy - 0.8), 5.0, 3.6, fc="#c9d4ec", ec=NAVY, lw=0.6))
ax.add_patch(Circle((dx - 2.4, dy + 2.4), 1.1, fc="none", ec=CRIMSON, lw=1.0))
for (u, v) in [(-0.5, 3.6), (1.6, 3.0), (3.2, 1.9)]:
    ax.add_patch(Rectangle((dx + u - 0.45, dy + v - 0.45), 0.9, 0.9, fc=CRIMSON, ec="none"))
txt(dx, 91, "Antigen-presenting cell", fontweight="bold")
txt(dx, 69.5, "antigen $A$ on\nMHC-I and MHC-II", color=CRIMSON)
arrow((18.6, 80), (28.5, 80))
txt(23.5, 82.2, "uptake", fontsize=6.5, style="italic")

# --- humoral branch
arrow((44.5, 84), (53.5, 92.5), col=FOREST)
gx, gy = 61, 94
ax.add_patch(Ellipse((gx, gy), 13, 7.5, fc=PALE_G, ec=FOREST, lw=0.9))
for (u, v) in [(-4, 0.8), (-1.8, -1.6), (0.3, 1.4), (2.5, -1.2), (4.3, 1.0), (-0.3, -0.2), (1.9, 0.4)]:
    ax.add_patch(Circle((gx + u, gy + v), 0.95, fc="white", ec=FOREST, lw=0.6))
txt(gx - 1, 100.2, "germinal centre $G$", color=FOREST)
arrow((67.8, 94), (73.0, 94), col=FOREST)
ax.add_patch(Circle((76.5, 94), 2.6, fc="white", ec=FOREST, lw=0.9))
ax.add_patch(Circle((76.5, 94), 1.2, fc=PALE_G, ec=FOREST, lw=0.5))
txt(76.5, 88.6, "memory\nB cells $B$", color=FOREST)
arrow((79.4, 94), (84.6, 94), col=FOREST)
ax.add_patch(Ellipse((88.5, 94), 7.2, 5.0, fc="white", ec=FOREST, lw=0.9))
for k in range(4):
    ax.plot([86.2, 90.8], [92.6 + 0.9 * k, 92.6 + 0.9 * k], color=FOREST, lw=0.4)
txt(89.5, 100.2, "plasma cells $P$", color=FOREST)
arrow((92.4, 94), (97.6, 94), col=FOREST)


def ylet(x, y, s=1.6, rot=0.0):
    c, sn = np.cos(rot), np.sin(rot)
    pts = [((0, -s), (0, 0)), ((0, 0), (-0.7 * s, 0.8 * s)), ((0, 0), (0.7 * s, 0.8 * s))]
    for (a, b) in pts:
        ax.plot([x + c * a[0] - sn * a[1], x + c * b[0] - sn * b[1]],
                [y + sn * a[0] + c * a[1], y + sn * b[0] + c * b[1]], color=FOREST, lw=1.1,
                solid_capstyle="round")


for (u, v, ro) in [(0, 0.5, 0.0), (3.6, 1.6, -0.4), (2.2, -2.4, 0.5), (5.6, -1.0, 0.2)]:
    ylet(100.5 + u, 94 + v, rot=ro)
txt(103.5, 100.2, "IgG $H$", color=FOREST)
txt(103.5, 88.0, "MSD IgG assay", fontsize=6.5, style="italic", color=FOREST)

# --- cellular branch
arrow((44.5, 76), (53.5, 66.5), col=NAVY)
for (u, v) in [(-2.4, 0.9), (0.2, -1.2), (2.6, 0.8)]:
    ax.add_patch(Circle((60 + u, 64.5 + v), 1.6, fc="white", ec=NAVY, lw=0.8))
txt(60, 58.6, "primed\nprecursors $M$", color=NAVY)
arrow((64.8, 64.5), (70.6, 64.5), col=NAVY)
for (u, v) in [(-2.6, 1.2), (0.0, -1.3), (2.6, 1.2)]:
    ax.add_patch(Circle((76 + u, 64.5 + v), 2.0, fc=PALE_B, ec=NAVY, lw=0.9))
    txt(76 + u, 64.5 + v, "CD8", fontsize=4.6, color=NAVY)
txt(76, 58.6, "effector CD8$^{+}$\nT cells $E$", color=NAVY)
txt(76, 70.4, "IFN-$\\gamma$ ELISpot", fontsize=6.5, style="italic", color=NAVY)
arrow((81.0, 64.5), (88.5, 64.5), col=NAVY)
txt(84.7, 66.6, "killing", fontsize=6.5, style="italic")
rng = np.random.default_rng(3)
tum = [(0, 0), (2.6, 1.2), (-2.4, 1.6), (1.0, 3.4), (-1.4, -2.4), (2.4, -1.8), (4.6, 0.2), (-0.6, 5.2),
       (3.6, 3.8), (-3.8, -0.6), (0.8, -4.2), (4.9, -3.0)]
for i, (u, v) in enumerate(tum):
    c = (96 + u, 64 + v)
    ax.add_patch(Circle(c, 1.45, fc=PALE_R, ec=CRIMSON, lw=0.7))
    ax.add_patch(Circle(c, 0.5, fc=CRIMSON, ec="none"))
    if i in (9, 4):
        ax.plot([c[0] - 1.0, c[0] + 1.0], [c[1] - 1.0, c[1] + 1.0], color=BLACK, lw=0.8)
        ax.plot([c[0] - 1.0, c[0] + 1.0], [c[1] + 1.0, c[1] - 1.0], color=BLACK, lw=0.8)
txt(97, 56.8, "tumour burden $X$", color=CRIMSON)

# --- reactogenicity
arrow((11, 66.2), (11, 59.6), col=BLACK)
box(1, 51.5, 38, 7.8, PALE_K, BLACK)
txt(20, 55.4, "innate sensing of RNA and lipid\n→ reactogenicity, CTCAE grade $\\mathsf{g}_t$")

# --- interval effect
box(113.5, 64, 29.5, 34, "white", BLACK)
txt(128.25, 94.2, "Interval $\\Delta$ between doses", fontweight="bold")
txt(128.25, 87.6, "longer $\\Delta$: memory pool $B(\\Delta)$\nhas matured further", color=FOREST)
txt(128.25, 81.3, "⇒ larger antibody boost", color=FOREST)
txt(128.25, 74.4, "longer $\\Delta$: precursor pool\n$M(\\Delta)$ has decayed", color=NAVY)
txt(128.25, 68.1, "⇒ smaller T-cell boost", color=NAVY)

ax.plot([1, 143], [48.6, 48.6], color=GREY, lw=0.6, ls=(0, (4, 3)))

# ============================== panel (b) ==============================
txt(1, 45.5, "(b) Abstraction used for weekly booster decisions", ha="left", fontweight="bold", fontsize=8)

box(1, 2, 34, 39.5, "white", BLACK)
txt(18, 38.4, "Observations at week $t$", fontweight="bold")
rows = [("ELISpot readout", "$C_t$"), ("IgG readout", "$H_t$"), ("tumour change", "$U_t$"),
        ("grade $\\geq3$ event", "$\\mathrm{Tox}_t$"), ("grade $\\geq2$ history", "$R_t$"),
        ("baseline risk", "$B_t$"), ("weeks since dose", "$S_t$"), ("doses received", "$N_t$")]
ys = np.linspace(34.0, 6.0, 8)
for (lab, sym), y in zip(rows, ys):
    txt(3.2, y, lab, ha="left")
    txt(25.5, y, "→", ha="center")
    txt(28.0, y, sym, ha="left", color=CRIMSON)
arrow((35.6, 22), (40.4, 22))

box(41, 2, 41, 39.5, "white", BLACK)
txt(61.5, 38.4, "Markov $k$-tree transition model", fontweight="bold")
names = ["C", "H", "U", "Tox", "R", "B", "S", "N"]
ny = np.linspace(33.0, 9.5, 8)
xl, xr = 51.5, 72.5
txt(xl, 36.0, "$t$", color=BLACK); txt(xr, 36.0, "$t+1$", color=BLACK)
for n, y in zip(names, ny):
    for x in (xl, xr):
        ax.add_patch(Circle((x, y), 1.45, fc="white", ec=BLACK, lw=0.7))
    lbl = "Tox" if n == "Tox" else n
    txt(xl - 3.6, y, lbl, fontsize=6.2, ha="right")
    ax.plot([xl + 1.5, xr - 1.5], [y, y], color=GREY, lw=0.5, ls=(0, (2, 2)))
pos = {n: y for n, y in zip(names, ny)}
for (child, par) in [("Tox", "B"), ("R", "B"), ("H", "S"), ("U", "N"), ("C", "N"), ("U", "C"), ("Tox", "R")]:
    ax.add_patch(FancyArrowPatch((xl + 1.4, pos[par]), (xr - 1.5, pos[child]), arrowstyle="-|>",
                                 mutation_scale=5, color=CRIMSON, lw=0.6, shrinkA=0, shrinkB=0))
txt(61.5, 5.0, "each node at $t+1$: own past, action $a_t$\nand at most $k$ parents; Dirichlet tables",
    fontsize=6.2)
arrow((82.6, 22), (87.4, 22))

box(88, 2, 26.5, 39.5, "white", BLACK)
txt(101.25, 38.4, "Safety screen", fontweight="bold")
txt(101.25, 30.0, "$\\mathrm{UCB}_{0.90}$ of\n$\\mathbb{P}(\\mathrm{Tox}_{t+1}=1\\mid\\mathbf{z}_t,a)$\n$\\leq\\delta$",
    color=CRIMSON)
ax.plot([90, 112.5], [21.5, 21.5], color=GREY, lw=0.5)
txt(101.25, 13.5, "backward induction\nover admissible actions\nmaximizes expected\nclinical utility")
arrow((115.1, 22), (119.9, 22))

box(120.5, 2, 22.5, 39.5, "white", BLACK)
txt(131.75, 38.4, "Action $a_t$", fontweight="bold")
for (lab, y, col) in [("defer", 30.0, BLACK), ("full booster", 22.0, NAVY), ("reduced booster", 14.0, FOREST)]:
    ax.add_patch(FancyBboxPatch((122.6, y - 2.6), 18.3, 5.2, boxstyle="round,pad=0,rounding_size=2.4",
                                fc="white", ec=col, lw=0.9))
    txt(131.75, y, lab, color=col)
txt(131.75, 6.0, "a booster adds\na new dose to $R$", fontsize=6.2, color=CRIMSON)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/Fig1_Mechanism_and_abstraction.png")
print("ok")
