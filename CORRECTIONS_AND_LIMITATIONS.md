# Corrections and implementation limitations

This note reconciles the frozen protocol with the final pre-submission audit.
It changes neither the frozen patient split nor any disease prediction.

## Protocol execution

The endpoint hierarchy, primary estimand, frozen ConvNeXt-Tiny representation,
split, calibration selection, and primary/fixed-sequence inference were
executed without deviation. Several prospectively registered
secondary/sensitivity analyses were omitted from the initial R7/R8 execution.
After the primary results had been observed, the executable registered analyses
were completed from the existing sealed probabilities and retained as
secondary evidence only. An extended fellow-eye quality specification was not
executed because its frozen description did not uniquely determine a design
matrix without new researcher discretion.

The completed state-matched unrelated-eye control did not support specificity
of the reliability improvement to the true within-patient pairing. Because
that control retrospectively matches on ground-truth bilateral state, it is
nondeployable and cannot establish a biological mechanism.

## Internal cross-fitting limitation

Disease-head outer assessment remained patient-isolated, and the final holdout
was completely isolated from selection. Inner meta-selection used pooled
out-of-fold logits. Those prediction features can contain a second-order
cross-fold dependency through disease-head training. This can affect strict
unbiasedness of internal selection estimates, but it cannot directly create
final-holdout leakage. Future replication should use fully nested meta-level
cross-fitting.

## Software provenance

The sealed reliability fits used scikit-learn 1.5.2 Ridge with realized
defaults `solver='auto'`, `tol=0.0001`, and `max_iter=None`.

## Scientific boundaries

Only BRSET was evaluated; no independent external validation was performed.
The findings may depend on the frozen ConvNeXt-Tiny representation, and another
model may exhibit a different failure structure. Negative log loss is
unbounded for confidently wrong probabilities and increases as the negative
log probability assigned to the true class. The observed improvement was
influenced by this high-loss tail, although the frozen diagnostics did not find
dependence on one patient or solely on the highest-loss 1%.
