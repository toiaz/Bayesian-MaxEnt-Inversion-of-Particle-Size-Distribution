# Fuel R1 Reproducibility Package

This repository release provides the processing and inversion materials used for the
Fuel revision of the Bayesian-MaxEnt PSD manuscript.

## Tagged Version

Use tag `v1.0.0-fuel-r1` for the reviewer-response reproducibility package.

## Scope

Included:

- Core processing and inversion code.
- Synthetic benchmark input datasets.
- Processed shale (V)SANS input used in the inversion.
- Key numerical inversion outputs and logs underlying the reported synthetic and
  shale results.
- Numerical evidence/cache output used for the synthetic benchmark diagnostics.

Not included:

- Final figure-layout or journal-formatting scripts.
- Manuscript drafts, reviewer-response files, and submission-package working files.
- Large intermediate outputs and unrelated historical test runs.

The public package is intended to reproduce the inversion results and numerical
diagnostics reported in the manuscript, rather than the exact final figure layout.

## Main Files

Processing and inversion code:

- `Code/MaxEnt_core_v3.2.1.py`
- `Code/generate_synthetic_sasv1.4.py`

Processed shale input:

- `Code/Test.dat`

Synthetic benchmark datasets:

- `Code/Synthetic_data/GUI_v1.4_20260301_145846-with-sf/`
- `Code/Synthetic_data/GUI_v1.4_20260301_152027-without-sf/`

Key synthetic numerical outputs:

- `Code/PSD_results/20260301_151900_SynData_Poisson-with-sf-with-grid/`
- `Code/PSD_results/20260301_230208_SynData_Poisson-with-sf-without-gridv1/`

Key shale numerical outputs:

- `Code/PSD_results/20260304_005237_Test-possion-hard sphere/`
- `Code/PSD_results/20260304_005457_Test-possion-unity/`

Each listed output folder contains the numerical PSD results and simulation log
used to report the manuscript diagnostics.
