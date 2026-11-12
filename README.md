# Safety-constrained circRNA cancer vaccine boosting with Markov k-tree transition models

This repository contains the code, data and results for the study *Safety-constrained optimization of circRNA cancer vaccine boosting schedules with Markov k-tree transition models and mechanistic priors*.

The framework schedules weekly booster decisions (defer, full booster, reduced booster) for a therapeutic circRNA cancer vaccine. It combines:

- a mechanistic immune-kinetic module (antigen expression, germinal-centre maturation, memory B cells, plasma cells, IgG, primed T-cell precursors, effector T cells and tumour burden), calibrated on individual-level human data from the PITCH study;
- a Markov k-tree transition model, in which each next-state variable depends on its own past, the action and at most k other variables, with the mechanistic module entering as Dirichlet pseudo-counts whose strength is selected by patient-grouped held-out likelihood;
- a chance constraint on grade ≥3 toxicity, enforced through a posterior upper credible bound;
- exact finite-horizon backward induction with variable elimination.

## Repository layout

| Folder | Content |
|---|---|
| `code/` | All Python source files |
| `data/` | PITCH antibody (MSD IgG) and T-cell (IFN-γ ELISpot) measurements with participant metadata |
| `results/` | Outputs used in the manuscript (CSV/JSON) and `numbers.txt`, which lists every number quoted in the text |
| `figures/` | Figures 1–9 of the manuscript |

## Code files

| File | Purpose |
|---|---|
| `mech.py` | Mechanistic immune-kinetic model (Euler integration) |
| `pitch_fit.py` | Calibration on long-interval PITCH participants and validation on held-out short-interval participants |
| `sim.py` | Virtual-patient simulator (heterogeneity, frailty, toxicity, missing visits) |
| `baselines.py` | Fixed schedules (three-weekly, prime-boost, extended) and the clinician-style rule |
| `ktree.py` | Discretization, Markov k-tree structure and prior-strength selection, credible toxicity bound, planner |
| `fqi.py` | Fitted Q-iteration comparator |
| `core.py` | Data generation, policy fitting and evaluation helpers |
| `run_all.py` | All simulation experiments (main comparison, frontier, ablation, sensitivity, training size, platform, calibration, prior misspecification, runtime) |
| `extra.py` | Interval effect, platform kinetics and case studies |
| `fig_background.py` | Figure 1 (schematic) |
| `figures.py` | Figures 2–9 |
| `report_numbers.py` | Writes `results/numbers.txt` |

## Requirements

Python 3.10 or later.

```
pip install -r requirements.txt
```

## Reproducing the results

Run from the repository root:

```
python code/pitch_fit.py        # calibration and held-out validation on PITCH data
python code/run_all.py 20       # all experiments, 20 independent runs (about 1 h on 2 cores)
python code/extra.py            # interval effect, platform kinetics, case studies
python code/fig_background.py   # Figure 1
python code/figures.py          # Figures 2-9
python code/report_numbers.py   # results/numbers.txt
```

All random seeds are fixed inside the scripts, so repeated runs give the same outputs.

## Data source

The files in `data/` were extracted from the public PITCH dataset (Payne et al., *Cell* 2021, doi:10.1016/j.cell.2021.10.011), released on Mendeley Data under the CC BY 4.0 licence (doi:10.17632/fyp26zjgmj.1). See `data/README.md`.

## Licence

The code is released under the MIT licence (see `LICENSE`). The PITCH data remain under their original CC BY 4.0 licence.
