# R5 Primary Estimand Definition

## Candidate C ambiguity adjudication

### Interpretation 1 — failure/loss prediction

A disease model produces a patient-out-of-sample probability. The observed label and that probability define eye-level loss. Separate reliability models estimate expected loss from a frozen own-eye information set, with or without fellow-eye information. The estimand asks whether the fellow eye improves prediction of realized loss.

This interpretation preserves the R4 scientific question. The disease probability is not changed by the reliability model, and the work remains failure analysis rather than bilateral disease fusion.

### Interpretation 2 — bilateral probability recalibration

A baseline calibrator transforms the own-eye disease output, while an augmented calibrator uses the fellow-eye output to generate a new disease probability. Improvement is measured directly as disease-prediction NLL reduction.

This is a bilateral late-fusion/calibration method. Even if labeled “reliability,” its operational purpose is to improve disease probabilities using multiple images. It therefore moves toward the spaces occupied by SureSight and related confidence-guided multi-image methods and weakens the frozen contribution boundary.

### Frozen choice

**Interpretation 1 is selected. Interpretation 2 is prohibited as a primary or confirmatory analysis.**

The endpoint calibrator is monocular: it maps an eye’s raw disease logit using development data and never uses the fellow eye. The reliability model predicts loss but does not alter the disease probability. Any bilateral recalibration performed later is exploratory, requires a protocol amendment, and cannot replace the frozen analysis.

## Notation

For patient \(p\) and eye \(i\in\{R,L\}\):

- \(y_{pi}\in\{0,1\}\) is the frozen endpoint label;
- \(p_{pi}\) is the calibrated probability generated without training on patient \(p\);
- \(\tilde p_{pi}=\min(1-10^{-6},\max(10^{-6},p_{pi}))\) is used only for finite logs;
- \(z_{pi}=\operatorname{logit}(\tilde p_{pi})\);
- \(L_{pi}=-[y_{pi}\log\tilde p_{pi}+(1-y_{pi})\log(1-\tilde p_{pi})]\).

Raw calibrated probabilities are retained. The number clipped is reported. Sensitivities use \(10^{-5}\) and \(10^{-7}\). At the primary clipping value, the numerical NLL ceiling is \(-\log(10^{-6})=13.815510557964274\).

## Frozen reliability information sets

### Baseline information set \(X^{(0)}_{pi}\)

The primary retrospective baseline contains only:

1. a cubic B-spline basis of the own-eye calibrated logit \(z_{pi}\), using 5 training-fold quantile knots, degree 3, linear extrapolation, and no bias column;
2. own-eye `image_field`, coded normal versus abnormal;
3. own-eye `focus`, coded normal, abnormal, or unevaluable;
4. laterality, coded right versus left;
5. eye-level acquisition stratum, Canon CR versus NIKON NF5050.

The spline transformer and standardization parameters are learned inside the relevant training fold. No outcome-driven feature selection is permitted.

Entropy, distance from 0.5, maximum class probability, and “confidence” are excluded because, for a binary calibrated probability, they are deterministic transforms of the probability/logit already represented flexibly. They are not independent uncertainty signals.

### Augmented information set \(X^{(1)}_{pi}\)

The primary augmented set is

\[
X^{(1)}_{pi}=X^{(0)}_{pi}+\text{cubic B-spline basis of }z_{p\bar i},
\]

where \(\bar i\) is the fellow eye. The fellow-eye spline uses the same frozen construction but is fit within the reliability-training fold.

The primary model does not simultaneously include fellow-eye logit, signed asymmetry, pair evidence, entropy, or confidence. Conditional on own-eye logit, fellow-eye logit is a one-to-one representation of oriented inter-eye logit contrast; adding redundant representations would create rank deficiency or misleading feature attribution.

Fellow-eye `image_field`, fellow-eye `focus`, and signed bilateral quality contrasts are reserved for a prespecified retrospective mechanistic sensitivity. They are not in the primary augmented information set.

### Inference-valid core sensitivity

A locked sensitivity removes human `image_field` and `focus` from both baseline and augmented sets. Only this version can inform a future deployment-oriented claim without a separately validated automated quality module.

## Frozen reliability-model form

Both baseline and augmented models are ridge regressions fitted to raw NLL:

\[
\widehat R_k(X)=\widehat E[L\mid X^{(k)}],\qquad k\in\{0,1\}.
\]

Categorical variables use fixed reference coding. Continuous spline columns are standardized with training-fold means and standard deviations. Ridge \(\alpha\) is selected from \(\{0,0.01,0.1,1,10\}\) by inner patient-level cross-validation minimizing the primary MSE; ties within \(10^{-12}\) choose the larger \(\alpha\). Intercepts and categorical reference coding are identical between models.

Predicted loss is deterministically truncated to \([0,13.815510557964274]\) before evaluation because this is the attainable range after the frozen numerical probability clipping. No truncation level is selected from predictions.

No ground-truth bilateral state, own-eye label, fellow-eye label, endpoint grade, or ground-truth-derived residual may enter a reliability predictor.

## Comparison measures considered

| Measure | Strength | Limitation for this question | R5 role |
|---|---|---|---|
| Mean squared error for predicted NLL | Elicits the conditional mean, matching \(E[L\mid X]\); responds to failure to predict high-loss eyes | Sensitive to the right tail and confident errors | **Primary** |
| Mean absolute error | More robust to the NLL tail | Elicits a conditional median rather than expected loss | Secondary sensitivity |
| Out-of-sample \(R^2\) | Scale-normalized | Can be negative or unstable when baseline MSE is small | Secondary descriptive |
| Explained deviance | Potentially distribution-aware | Requires a defensible likelihood for the NLL target that is not available pre-prediction | Not selected |
| Rank correlation/discrimination | Tests ordering of risk | Does not assess magnitude or calibration of predicted loss | Secondary sensitivity |
| Error-detection AUPRC/AUROC | Clinically familiar secondary ranking summaries | Threshold-dependent error target and different question from continuous loss prediction | Secondary only |

NLL’s right skew is not removed or winsorized in the primary analysis. Patient-cluster bootstrap intervals and MAE sensitivity expose tail dependence.

## Primary risk and estimand

Let \(H\) be the untouched clean-bilateral final holdout containing \(N_H\) patients. For reliability model \(k\):

\[
\operatorname{Risk}_k=
\frac{1}{N_H}\sum_{p\in H}\frac{1}{2}
\sum_{i\in\{R,L\}}
(L_{pi}-\widehat R_k(X^{(k)}_{pi}))^2.
\]

The primary estimand is

\[
\Delta_R=\operatorname{Risk}_0-\operatorname{Risk}_1.
\]

Positive \(\Delta_R\) consistently means that fellow-eye logit information improves held-out prediction of own-eye NLL. Every patient has weight one, each eye has weight one-half within patient, and uncertainty is estimated by resampling patients with both eyes together.

The target population is the v1.0.2 clean bilateral cohort defined in `R6_COHORT_FREEZE.md`. The primary endpoint is drusen. Confirmatory endpoints use the identical formula and analytical specification.

## Secondary estimands

1. \(\Delta_{MAE}=MAE_0-MAE_1\) for predicted NLL.
2. Relative MSE improvement \(\Delta_R/\operatorname{Risk}_0\).
3. Difference in patient-clustered Spearman association between predicted and observed NLL.
4. The complete Candidate C analysis with Brier loss as the target.
5. Inference-valid core contrast excluding human quality.
6. Retrospective augmented-quality contrast adding fellow-eye quality to both a clearly labeled extended baseline and extended fellow-eye model.

None may replace the primary metric after results are seen.

## Tautology controls

- Own-eye logit is in both models; only fellow-eye logit differs.
- The outcome label never enters a predictor.
- Bilateral state is used only for retrospective stratification or mechanistic models.
- Direct \(A\)-versus-NLL association is not the primary reliability evidence.
- The disease calibrator is monocular.
- Reliability models are fit on development out-of-fold losses and evaluated on patients unused in their fitting.
- A matched unrelated-eye negative control tests whether any gain is pair-specific.
- The own-eye spline is mandatory; a misspecified linear own-eye term cannot be used to manufacture fellow-eye gain.

## Epistemic uncertainty

No epistemic estimator is part of the primary or confirmatory information sets. R6 separately freezes an empty epistemic candidate set. Any future ensemble or stochastic-inference quantity requires a pre-holdout protocol amendment and remains secondary.
