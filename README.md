# BIOL2595 Final Project

This project investigates whether low-cost, non-imaging clinical and demographic data can be used to predict time to amyloid positivity in ADNI participants. The workflow focuses on building a clean longitudinal dataset, defining amyloid conversion and censoring structure, and preparing the data for survival XGBoost modeling.

## Files

- `Table_Builder.py` — merges the main ADNI tables into a single longitudinal analysis table and assigns visit dates to each row.
- `AMY_Builder` — creates a subject-level Berkeley amyloid summary table with first observed amyloid-positive date information.
- `FINAL_BUILDER` — adds time-to-amyloid-positivity and censoring columns to the merged table using visit dates and Berkeley amyloid summary data.
- `XGBoost_attempt.py` — prepares the final analytic sample and fits a survival XGBoost model using censored time-to-event data.
- `Table1_builder` — generates baseline descriptive statistics and Table 1 files for the final analytic sample.

## Current Outputs

- merged longitudinal dataset with visit-level dates
- subject-level amyloid positivity summary table
- final survival modeling table with `TTAMY` and `AMY_EVENT`
- survival XGBoost input and prediction files
- manuscript-style Table 1 summary files
