# R6 Calibration and Uncertainty Registry

## Calibration target

Calibration is monocular and endpoint-specific. It maps an eye’s raw disease logit to a calibrated probability using only development patients. It never uses fellow-eye outputs, biological state, quality labels, or final-holdout labels.

## Registered calibration candidates

### C1 — binary temperature scaling

\[
p=\operatorname{sigmoid}(z_{raw}/T),\qquad T>0.
\]

- one fitted parameter;
- optimize NLL in log-temperature space;
- bound \(\log T\in[\log 0.05,\log 20]\);
- strictly monotonic and does not change ranking;
- no intercept, so it cannot correct a prevalence offset.

### C2 — Platt/logistic calibration

\[
p=\operatorname{sigmoid}(a z_{raw}+b),\qquad a>0.
\]

- two fitted parameters;
- optimize NLL with \(\log a\in[\log 0.05,\log 20]\) and \(b\in[-10,10]\);
- strictly monotonic because \(a>0\);
- can correct both scale and intercept.

The small two-method registry is appropriate for binary outputs and avoids an open calibration tournament. Isotonic regression, beta calibration, splines, and fellow-eye calibration are excluded. Isotonic is especially avoided because it adds a flexible step function, may create ranking ties, and would add a method-selection degree.

Primary methodological reference: [scikit-learn probability calibration documentation](https://scikit-learn.org/stable/modules/calibration.html), which describes sigmoid/Platt calibration, temperature scaling, and the need for calibration data independent of base-model fitting.

## Calibration selection and refit

1. Within each outer-development training cohort, generate inner patient-OOF raw logits for each disease-head \(C\).
2. For an inner assessment fold, fit the candidate calibrator on the other inner folds’ OOF logits and apply it to the held-out inner fold.
3. Select the \(C\)+calibrator combination minimizing patient-mean calibrated NLL across all inner assessment folds.
4. Exact NLL ties within \(10^{-12}\) prefer stronger head regularization, then temperature scaling.
5. For an outer assessment fold, refit the chosen calibrator on all inner OOF logits from the outer-training cohort and apply it to the outer-fold raw logits.
6. After nested development assessment, choose the modal calibrator identity and \(C\) separately for each endpoint; deterministic ties use the rule above.
7. For the final stack, fit that calibrator identity on full-development OOF logits and labels. Apply its locked parameters to final-holdout raw logits.
8. Confirmatory endpoints use the identical two-method registry and nested selection algorithm on their own development OOF logits; they cannot add a method or alter the criterion.

No calibrator may be refitted or corrected using the final holdout.

## Numerical clipping

- Retain the unmodified calibrated probability.
- For log/logit calculations only, clip to \([10^{-6},1-10^{-6}]\).
- Report endpoint/partition clipping counts.
- Repeat reported NLL/logit-derived sensitivities at \(10^{-5}\) and \(10^{-7}\).
- Do not select a clipping value by performance.

## Calibration reporting

- NLL and Brier;
- calibration intercept and slope;
- reliability diagram using 10 equal-frequency bin edges learned from development OOF probabilities and then fixed;
- ECE only when every required final-holdout bin contains at least 30 eyes;
- no holdout recalibration.

## Deterministic confidence

The calibrated probability/logit is the sole primary monocular confidence representation. Entropy, \(|p-0.5|\), maximum probability, margin, and “confidence” are deterministic functions of the same binary probability. They may be plotted descriptively but are not additional predictors or uncertainty methods.

## Epistemic uncertainty

**Registered epistemic candidate set: empty.**

No MC dropout, ensemble dispersion, stochastic depth sampling, test-time augmentation variance, or other epistemic score is part of the primary, confirmatory, or final-holdout protocol.

Reasons:

- the selected official ConvNeXt-Tiny has stochastic-depth blocks but no ordinary dropout layer supporting a standard MC-dropout procedure;
- a small ensemble would multiply compute and alter the base predictor;
- the scientific question can be answered by incremental fellow-eye information beyond the full calibrated own-eye logit;
- adding an uncertainty tournament would reintroduce avoidable degrees of freedom.

An epistemic method can only be introduced by a prospective protocol amendment before any final-holdout access and would remain secondary with independent evaluation. It cannot rescue an unfavorable primary result.
