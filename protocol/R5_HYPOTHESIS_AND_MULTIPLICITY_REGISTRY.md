# R5 Hypothesis and Multiplicity Registry

## Common definitions

- Primary endpoint: `drusen`.
- Confirmatory endpoints, in order: `increased_cup_disc`, then `diabetic_retinopathy`.
- Primary estimand: \(\Delta_R=Risk_0-Risk_1\) from `R5_PRIMARY_ESTIMAND_DEFINITION.md`.
- Unit of analysis and resampling: patient.
- Primary interval: patient-cluster BCa bootstrap.
- All disease and reliability specifications are frozen before final-holdout evaluation.
- Ground-truth state is never an inference-time predictor.

## H1 — primary incremental reliability hypothesis

**Population:** untouched clean-bilateral drusen holdout.
**Null:** \(H_{1,0}:\Delta_R\leq0\).
**Alternative:** \(H_{1,A}:\Delta_R>0\).
**Direction:** positive values favor incremental fellow-eye information.
**Test:** one-sided 95% BCa lower confidence bound; H1 is supported only if the lower bound is greater than zero. A two-sided 95% BCa interval is always reported.
**Inference:** 10,000 patient-cluster bootstrap resamples with both eyes retained together and seed 20261001.
**Interpretation:** this tests prediction of own-eye NLL, not disease-probability fusion.

## H2 — four-state heterogeneity

**Status:** prespecified secondary mechanistic hypothesis. It is not an inference-time model feature and is not required for H1.
**Null:** \(H_{2,0}:\Delta_{00}=\Delta_{11}=\Delta_{10}=\Delta_{01}\).
**Alternative:** at least one state-specific incremental value differs.
**Estimand:** state-conditional \(\Delta_s=E[d_{pi}\mid S_p=s]\), where \(d_{pi}\) is the eye-level squared-error advantage of the augmented reliability model.
**Test:** omnibus state term in the frozen clustered GEE plus state-stratified patient bootstrap estimates.
**Direction:** two-sided heterogeneity; no state is assumed symmetric or superior.
**Use:** four states remain separately reported even if the omnibus test is unsupported.

## H3 — quality modification

**Status:** prespecified secondary mechanistic family. Human quality labels make it retrospective, not deployable.
**H3a null:** incremental reliability contribution does not differ by own-eye `image_field` status.
**H3b null:** incremental reliability contribution does not differ by own-eye `focus` status among evaluable focus records.
**Alternative:** the corresponding quality-specific incremental value differs.
**Test:** separate quality terms in the frozen clustered GEE; H3a and H3b are Holm-adjusted as a two-test family.
**Direction:** two-sided.
**Restriction:** `image_field` and `focus` are never collapsed. The two invalid focus values are unevaluable. Fellow-eye and bilateral quality contrasts are secondary mechanistic sensitivities, not H3 replacements.

If a quality stratum produces a non-estimable interaction or bootstrap failure because of realized sparsity, that component is reported as not estimable; it is not pooled with another quality variable.

## H4 — increased cup-disc confirmatory replication

**Population:** untouched clean-bilateral `increased_cup_disc` holdout.
**Null:** \(H_{4,0}:\Delta_R\leq0\).
**Alternative:** \(H_{4,A}:\Delta_R>0\).
**Specification:** identical information sets, spline construction, candidate grids, selection algorithms, primary MSE risk, clustering, clipping, and inference used for H1. Disease-head, calibration, and reliability coefficients/hyperparameters are selected and fitted separately for this endpoint using development data only; no candidate, feature, or criterion changes by endpoint.
**Interpretation:** replication in a structural/anatomical finding; never a glaucoma claim.

## H5 — diabetic-retinopathy confirmatory replication

**Population:** untouched clean-bilateral diabetic-retinopathy holdout.
**Null:** \(H_{5,0}:\Delta_R\leq0\).
**Alternative:** \(H_{5,A}:\Delta_R>0\).
**Specification:** identical candidate sets, selection algorithms, information sets, outcome, and inference to H1 and H4, with endpoint-specific development-only fitting.
**Interpretation:** replication in a more bilaterally concordant microvascular endpoint. Sparse 10/01 estimates remain separate and may be imprecise.

## Multiplicity architecture

### Primary/generalization family: fixed sequence

The confirmatory reliability family uses a fixed-sequence gate at one-sided \(\alpha=0.05\):

1. test H1 (`drusen`);
2. only if H1 is rejected, test H4 (`increased_cup_disc`);
3. only if H4 is rejected, test H5 (`diabetic_retinopathy`).

Testing stops at the first failure. Later endpoint estimates and two-sided 95% intervals are still reported but are descriptive rather than confirmatory. The order is scientifically justified by the primary endpoint decision followed by the better-powered structural confirmatory endpoint and then the sparse-directional DR regime. No endpoint can be reordered after predictions.

### Mechanistic family

H2, H3a, and H3b form a separately labeled mechanistic family. Holm adjustment controls its three p-values at two-sided \(\alpha=0.05\). These tests do not recycle unused alpha from the primary/generalization sequence and cannot establish H1.

### Secondary outcomes

Brier-target results, MAE, relative MSE, rank measures, error-detection AUPRC/AUROC, AURC, selective risk, calibration diagnostics, inference-valid core analysis, random-eye negative control, and acquisition-stratum sensitivities are secondary. They do not enter the fixed sequence and cannot rescue a failed H1.

## Frozen biological-state use

The exact R0 states are:

| Endpoint | 00 | 11 | 10 | 01 |
|---|---:|---:|---:|---:|
| `drusen` | 6,002 | 984 | 363 | 346 |
| `increased_cup_disc` | 5,829 | 1,171 | 399 | 296 |
| `diabetic_retinopathy` | 7,153 | 440 | 60 | 42 |

Mechanistic signed-asymmetry model:

\[
A_p=\beta_0+f(E_p)+\beta_S S_p+\beta_{Fm}\bar Q^F_p+\beta_{Fd}\Delta Q^F_p+
\beta_{Zm}\bar Q^Z_p+\beta_{Zd}\Delta Q^Z_p+\beta_C C_p+
S_p\!:\!\Delta Q^F_p+S_p\!:\!\Delta Q^Z_p+
S_p\!:\!I(C_p=\mathrm{Nikon/Nikon})+\epsilon_p,
\]

where \(F\) denotes image field, \(Z\) denotes focus, means are bilateral quality levels, differences are right minus left, and \(f\) is the frozen 5-knot cubic spline for pair evidence. This model is descriptive. Acquisition interaction contrasts compare Nikon/Nikon with Canon/Canon; the eight mixed pairs retain a main indicator but are omitted from interaction contrasts because their cell support is insufficient.

Laterality is not a removable pair-level covariate: it is encoded by the fixed right-minus-left outcome, signed quality differences, and distinct 10/01 states. Prespecified descriptive orientation contrasts are the mean signed asymmetry in 00 and 11, and the 10-versus-01 difference in oriented discordant margin. These are reported with cluster-aware intervals under H2 but do not create extra rejection claims. OD/OS exchangeability is not assumed.

The complete focus model uses the 7,693 R0-evaluable pairs expected before splitting. A field-only model retains all 7,695 pairs. Acquisition is Canon/Canon, Nikon/Nikon, or mixed; the mixed group is described but not used for a within-patient cross-device claim.

## Threshold-dependent secondary outcomes

For secondary classification-error outcomes, the calibrated disease threshold is fixed at 0.5. Risk–coverage summaries use coverages 90%, 80%, and 70%. ECE uses 10 development-defined equal-frequency bins; holdout uses those fixed edges, and ECE is omitted if any required bin has fewer than 30 holdout eyes. These rules cannot be changed after viewing results.
