# R6 Statistical Inference Freeze

## Inferential unit

The patient is the sampling, splitting, resampling, and confidence-interval unit. Two eyes are correlated observations and are never counted as independent patients.

For the primary estimand, define each patient’s contribution:

\[
d_p=\frac{1}{2}\sum_{i\in\{R,L\}}
\left[(L_{pi}-\widehat R_0(X^{(0)}_{pi}))^2-
(L_{pi}-\widehat R_1(X^{(1)}_{pi}))^2\right].
\]

Then \(\widehat\Delta_R=N_H^{-1}\sum_{p\in H}d_p\).

## Primary bootstrap

| Item | Frozen choice |
|---|---|
| Resampling unit | Patient, retaining both eyes and all derived fields |
| Replicates | 10,000 |
| Seed | 20261001 |
| Interval | Bias-corrected and accelerated (BCa) |
| Primary directional decision | One-sided 95% lower BCa bound for \(\Delta_R\) |
| Required reporting | Two-sided 95% BCa interval plus point estimate and risks |
| Training refit within bootstrap | No; inference is conditional on the locked fitted pipeline |

BCa is selected because the patient contributions may be skewed by high NLL and because bias/acceleration correction is preferable to an unadjusted percentile interval for this smooth mean-difference statistic. The jackknife acceleration calculation deletes one patient at a time, retaining the other patients’ paired-eye contributions.

If BCa is numerically undefined because all bootstrap/jackknife statistics are identical, report the point mass and a percentile interval as a prespecified numerical fallback, label the fallback, and do not make a significance claim from it.

Monte Carlo adequacy is checked by repeating interval calculation with seeds 20261011 and 20261021 on the same fixed predictions. This checks numerical stability only; it does not create extra inferential analyses. Report bound variation to four decimal places.

## Four-state handling

- Overall H1 bootstrap samples patients from the holdout without state stratification, preserving the target population mixture in expectation.
- State-conditional estimates resample patients within each of 00, 11, 10, and 01.
- H2 reports every state denominator and state-specific \(\Delta_s\).
- No state is collapsed to make an interval computable.
- If a state lacks sufficient unique patient contributions for stable BCa acceleration, report percentile and exact denominator descriptively; it cannot support a confirmatory state claim.

## Clustered GEE sensitivity

The eye-level outcome is

\[
d_{pi}=(L_{pi}-\widehat R_0)^2-(L_{pi}-\widehat R_1)^2.
\]

Use Gaussian family, identity link, exchangeable working correlation, patient cluster, and robust sandwich standard errors.

Frozen covariates are:

- state indicators for 11, 10, and 01 with 00 reference;
- own-eye `image_field` abnormal indicator;
- own-eye `focus` abnormal indicator among valid focus plus an unevaluable indicator in the overall model;
- right-eye indicator;
- Nikon indicator with Canon reference;
- mixed-camera-pair indicator.

No automated variable selection is allowed. H2 tests the three state coefficients jointly. H3a tests `image_field`; H3b tests valid `focus`. State×quality interactions belong to the separate pair-level mechanistic model, not this primary GEE, unless reported as the frozen mechanistic sensitivity.

The GEE is a sensitivity analysis and cannot override the bootstrap primary result.

## Mechanistic signed-asymmetry inference

The pair-level outcome \(A_p=z_{pR}-z_{pL}\) has one row per patient. The frozen regression in the hypothesis registry uses:

- a 5-quantile-knot cubic spline for pair evidence \(E_p\);
- four-state indicators;
- bilateral mean and signed difference for `image_field`;
- bilateral mean and signed difference for `focus`;
- Canon/Canon, Nikon/Nikon, mixed acquisition category;
- state×signed-field-difference and state×signed-focus-difference interactions;
- state×Nikon/Nikon interactions with Canon/Canon reference; mixed pairs receive a main indicator but no interaction contrast.

Primary uncertainty for mechanistic contrasts uses patient bootstrap; HC3 heteroskedasticity-robust intervals are a model-based sensitivity. The full focus model is complete-case for valid focus. Biological, quality, and acquisition block contributions use cross-fitted squared-error reduction and are described as order-dependent associations, with reverse-order sensitivity.

No coefficient is causal. Camera coefficients are acquisition-stratum associations. A ground-truth-conditioned residual is not deployable.

Laterality is evaluated through the fixed right-minus-left orientation, separate 10 and 01 coefficients, signed quality contrasts, mean signed asymmetry in 00/11, and the 10-versus-01 oriented-margin contrast. No analysis swaps or pools eyes to impose exchangeability.

## Multiplicity and confidence intervals

- H1/H4/H5 follow the fixed sequence in R5.
- H2/H3a/H3b use Holm within the mechanistic family.
- Secondary metrics report unadjusted intervals explicitly labeled secondary.
- All confirmatory estimates include two-sided 95% intervals even when the hypothesis direction is positive.

## Missingness and non-estimability

- No endpoint labels are missing in the frozen cohort.
- Invalid focus is not imputed as normal or abnormal.
- Age is not an inferential covariate.
- A failed model fit, singular design, empty subgroup, or unstable bootstrap is reported as not estimable; predictors/states are not combined post hoc.

## Scope of uncertainty

The primary bootstrap conditions on this dataset, frozen partition, encoder, and fitted development stack. It captures patient sampling variability in the final holdout but not alternate-pretraining, training-seed, model-family, or external-domain variability. Claims must remain conditional on the tested system and BRSET acquisition strata.
