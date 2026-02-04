# Threshold Completeness

Experiments and plots for additive coverage by smooth numbers; includes code to compute coverage, residue diagnostics, and threshold scans.

## What this repo contains
- Source: `tc/` and `scripts/`
- Results: selected CSVs and plots
- Status: research code; claims in README are conservative / non-assertive

## Getting started
1. Create a virtual environment
2. pip install -r requirements.txt
3. Run `scripts\run_experiment.py` or `scripts\make_all.py` on small n to verify

## Reproducibility notes
- Exact seeds and grid parameters are recorded in CSVs.
- Large runs require significant RAM/CPU; start with n=1e6.
