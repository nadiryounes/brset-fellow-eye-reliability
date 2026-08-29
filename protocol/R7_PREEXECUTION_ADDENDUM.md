# R7 Pre-Execution Terminology Addendum

Date: 2026-08-29
Timing: created before any R7 weight download, embedding generation, model fitting, or prediction

## Clarification

The frozen R6 epistemic-uncertainty candidate registry is empty. The primary own-eye reliability baseline therefore controls for the calibrated own-eye probability/logit and the other prespecified baseline covariates; it does not control for an independently estimated epistemic-uncertainty quantity.

Throughout R7 and subsequent reporting, the permitted terms are:

- calibrated own-eye probability;
- calibrated probability confidence;
- probability-based reliability information;
- deterministic probability-derived confidence.

The project will not describe the primary baseline as controlling for:

- epistemic uncertainty;
- MC-dropout uncertainty;
- ensemble uncertainty;
- full predictive uncertainty.

Entropy, distance from 0.5, maximum binary probability, and similar confidence summaries are deterministic transformations of the same calibrated binary probability. They are not separate uncertainty estimators and are not added as duplicate predictors.

## Effect on the frozen protocol

This addendum is a terminology correction only. It does not change:

- the scientific question;
- the Candidate C failure/loss-prediction interpretation;
- the baseline or augmented information sets;
- the primary estimand;
- any endpoint, cohort, split, model, preprocessing, calibration, metric, hypothesis, multiplicity, or inferential rule;
- the empty epistemic-uncertainty registry.

No R0-R6 document is modified. If later prose uses the broader phrase “monocular uncertainty,” it must be replaced or immediately bounded to calibrated probability-derived confidence.

## Pre-execution integrity

`Model weights downloaded in R7 before this addendum: NO`

`Embeddings generated in R7 before this addendum: NO`

`Predictions generated in R7 before this addendum: NO`

`Scientific estimand changed: NO`
