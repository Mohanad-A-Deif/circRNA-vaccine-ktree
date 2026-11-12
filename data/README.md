# PITCH data

Extracted from: Payne RP, et al. Immunogenicity of standard and extended dosing intervals of BNT162b2 mRNA vaccine. *Cell* 2021;184:5699-5714. doi:10.1016/j.cell.2021.10.011.
Dataset: Mendeley Data, version 1, doi:10.17632/fyp26zjgmj.1, licence CC BY 4.0.

| File | Content |
|---|---|
| `pitch_meta.csv` | Participant identifier, centre, prior-infection status, dosing interval (days and LONG/SHORT group), vaccine type and month of each dose |
| `pitch_msd.csv` | Spike, nucleocapsid and RBD IgG (MSD assay) by participant and visit |
| `pitch_elispot.csv` | Spike S1 and S2 IFN-γ ELISpot (SFU) by participant and visit |

Only infection-naive participants with samples after the first dose are used in the analysis (see `code/pitch_fit.py`).
