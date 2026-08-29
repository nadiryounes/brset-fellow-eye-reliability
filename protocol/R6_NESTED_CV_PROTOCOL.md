# R6 Nested Patient-Level CV Protocol

## Scope

All model, regularization, calibration, and reliability-model choices occur inside the frozen 75% development cohort. The final 25% holdout has no role until the full stack is locked.

No prediction is generated in R6. This document specifies the future procedure.

## Fold architecture

| Layer | Frozen choice |
|---|---|
| Development outer folds | 5 |
| Inner folds within each outer-training cohort | 3 |
| Splitting unit | Patient |
| Outer seed | 20260901 |
| Inner seed | 20260902 plus outer-fold index |
| Training/solver seed base | 20260903 |
| Primary stratification | Drusen state 00/11/10/01 |
| Secondary balance audit | Canon/Nikon/mixed acquisition; confirmatory states; image field; focus |

Use shuffled stratified K-fold assignment on one row per patient. Camera is audited but not added to the CV stratum because the mixed group is sparse; overconstraining state×camera cells is prohibited. Every fold must contain all four drusen states. If a library implementation would place a patient more than once, stop; grouping assertions override convenience.

For exact reproducibility, patient order before splitting is ascending numeric `patient_id`, with string fallback. Random seeds and library versions are recorded.

## Frozen feature-extraction rule

The ConvNeXt encoder and preprocessing are entirely fixed and pretrained outside BRSET. Development images may therefore be encoded once after R7 authorization, with no labels supplied to the encoder. This does not relax patient-level head/calibration/reliability folds.

Final-holdout embeddings are not generated until the disease-head, calibration-method identity, reliability specification, ridge alpha, thresholds, and analysis code are locked. No embedding is generated in R6.

## Outer-fold disease procedure

For outer fold \(o\):

1. Set outer assessment patients aside.
2. Within the remaining patients, construct three inner folds stratified by drusen state.
3. For each logistic-head \(C\in\{0.01,0.1,1,10\}\), fit the scaler and head on two inner folds and generate raw logits for the third. Repeat to obtain inner patient-OOF logits.
4. Cross-fit both calibration candidates across those inner OOF logits, so no calibrator is assessed on a patient used to fit that calibrator.
5. Select the \(C\)+calibrator pair minimizing patient-mean calibrated drusen NLL. Ties within \(10^{-12}\) choose smaller \(C\), then temperature scaling.
6. Refit the scaler and selected logistic head on all outer-training patients.
7. Fit the selected calibrator on all inner OOF logits/labels from the outer-training patients.
8. Apply the locked outer-fold disease stack once to the outer assessment patients.

This yields one calibrated development OOF probability per eye without that patient contributing to the scaler, logistic head, or calibrator used for it.

## Outer-fold reliability procedure

Within each outer-training cohort:

1. Use its inner OOF calibrated probabilities to construct NLL targets and frozen baseline/augmented reliability features.
2. Select ridge \(\alpha\in\{0,0.01,0.1,1,10\}\) separately for baseline and augmented models by three-fold patient CV minimizing patient-mean MSE. Ties within \(10^{-12}\) choose larger \(\alpha\).
3. Fit the two reliability models on all outer-training inner-OOF examples.
4. Apply them to the calibrated probabilities/features of the outer assessment patients.

This yields development OOF predicted losses for both reliability models. The outer assessment patient is absent from disease-head fitting, calibration fitting, and reliability-model fitting.

Human quality labels may enter the frozen retrospective baseline only after the disease probability has been generated. They never enter the image encoder or disease head.

## Global development choices

After completing all five outer folds:

- final logistic \(C\), separately by endpoint: modal outer-fold winner; tie chooses smaller \(C\);
- final calibration method identity, separately by endpoint: modal outer-fold winner; tie chooses temperature scaling;
- final baseline/augmented ridge alphas, separately by endpoint and reliability model: modal outer-fold winners; tie chooses larger \(\alpha\);
- preprocessing, features, splines, and categorical variables are already fixed and cannot be selected;
- every endpoint uses the identical grids, criteria, folds, and tie rules; endpoint-specific development labels determine only that endpoint's coefficients and registered-grid selections.

Each endpoint retains its own development-selected method/hyperparameter identities and fitted coefficients. The confirmatory endpoints cannot change registries, criteria, folds, or tie rules after their development selections are complete, and no candidate is reselected using final-holdout performance.

## Final development refit before holdout

After the choices above and all analysis code are sealed:

1. Generate five-fold development OOF raw logits using each endpoint's development-selected logistic \(C\).
2. Fit the frozen calibration-method identity to those endpoint-specific OOF logits/labels.
3. Convert development OOF logits to calibrated OOF probabilities and losses.
4. Fit the frozen baseline and augmented reliability models with their selected alphas on those OOF losses/features.
5. Fit the endpoint-specific scaler/logistic head on all development patients.
6. Only then encode/predict the final holdout once, calibrate with development parameters, and apply the development-fitted reliability models.

No coefficient, knot, scaling value, threshold, calibration parameter, or loss model is refit on final-holdout data.

## Prediction provenance fields required in R7

Every development OOF and final-holdout prediction record must internally carry:

- patient ID and eye;
- partition and outer/inner fold provenance;
- endpoint;
- encoder weight identifier and preprocessing hash;
- logistic \(C\), scaler fit cohort, and head fit cohort;
- calibration method and fit cohort;
- clipping indicator;
- reliability specification and fit cohort.

These provenance files remain DUA-protected and are not part of the public manuscript package.

## Thresholds and secondary metrics

- binary disease threshold: 0.5 after calibration;
- risk–coverage points: 90%, 80%, 70%;
- ECE: 10 equal-frequency edges from full-development OOF probabilities, fixed for holdout, with the minimum-bin rule in R5;
- no threshold is optimized for accuracy, F1, Youden index, or favorable error counts.

## Leakage assertions

- no patient crosses development/holdout or any assessment/training fold;
- scaler fits are fold-specific;
- calibrator inputs are generated without the calibration patient being used by the base head;
- reliability targets/features are OOF relative to the base head and calibrator;
- the final holdout cannot choose \(C\), calibration, uncertainty, ridge alpha, spline form, variables, thresholds, or plots;
- confirmatory endpoints cannot choose a different analytical specification;
- endpoint labels never enter the image encoder.
