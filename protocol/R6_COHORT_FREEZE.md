# R6 Cohort Freeze

## Primary cohort

The primary modeling, mechanistic, reliability, and final-holdout population is the R0 clean bilateral cohort:

> one patient with exactly two metadata rows, exactly one `exam_eye=1` right/OD image and exactly one `exam_eye=2` left/OS image, valid binary labels for all three frozen endpoints, and both image files present.

R0 contains **7,695** such patients/pairs. All three endpoint labels are complete in this cohort. The patient is the only partition key.

## Inclusion/exclusion decisions

| Structure | R0 evidence | Frozen decision |
|---|---:|---|
| Exactly one OD + one OS | 7,695 patients | Include in primary cohort |
| Single-eye patients | 815 patients | Exclude from all primary and confirmatory model development/evaluation |
| Duplicate-eye patients | 25 patients; 33 duplicated patient-eye combinations | Exclude the entire patient from model development/evaluation |
| Patients with >2 images | 14 patients, potentially overlapping duplicate-eye category | Exclude the entire patient; do not choose images |
| Multi-camera patients | 10 overall; 8 clean cross-camera pairs | Include only when they otherwise satisfy the clean-pair rule; keep as one patient and code acquisition pair as mixed |
| Missing image files | 0 in R0 | Stop on any future discrepancy |
| Exact duplicate image bytes | 0 in R0 | Stop/reconcile before training if future integrity check changes |
| Invalid `focus=0` | 2 image records | Retain patient/endpoint; code focus unevaluable. Exclude only from focus-specific complete-case mechanistic estimates. |
| Invalid `optic_disc='bv'` | 1 record | Retain because optic-disc anatomy is not a primary predictor; flag only |
| Age missing/inconsistent | Substantial missingness and some inconsistency in R0 | Do not exclude; age is balance-description only |
| Sex inconsistency | R0-audited metadata issue if present | Do not use sex as predictor; describe balance using patient-consistent value or unknown |

Categories overlap; excluded counts are never summed without patient-level reconciliation.

## Why single-eye and ambiguous multi-image patients are excluded

The primary question requires a true fellow eye. Allowing single-eye patients only in disease-head training would change the development spectrum relative to the paired target population and create a hidden source of model selection. Selecting among repeated images would create an outcome-adaptive degree of freedom. The clean-pair cohort is sufficiently large, so all non-clean patients are excluded from this protocol, including base-head development.

They remain in the untouched raw release and may support a separately registered future transportability study.

## Multi-camera handling

The eight clean cross-camera pairs stay eligible and are assigned as indivisible patients. Acquisition-pair categories are:

- Canon/Canon;
- Nikon/Nikon;
- mixed.

Mixed pairs are retained to preserve the audited cohort, but their count cannot identify a within-patient cross-device relationship. Sensitivity analysis excludes them. Camera language remains acquisition/domain language only.

## Quality handling

- `image_field`: 1 normal, 2 abnormal; no automatic composite.
- `focus`: 1 normal, 2 abnormal, 0 invalid/unevaluable.
- primary retrospective reliability baseline uses own-eye field and focus;
- inference-valid core omits both human labels;
- mechanistic focus model uses evaluable focus only;
- illumination and artifacts are exploratory.

No quality field is used to exclude an otherwise eligible patient. Excluding poor-quality images would remove a mechanism the study intends to examine.

## Metadata anomalies

Camera, sex, age, diabetes history, and other patient-level inconsistencies do not trigger exclusion unless they prevent patient identity or laterality from being established. Patient identity and eye mapping were audited in R0. Free text is never used. Raw values are not corrected.

## Partition population

The final partition contains exactly the 7,695 clean-pair patient IDs and two columns only:

`patient_id,partition`

No image ID, path, label, quality value, camera, age, or sex is copied to the manifest. Aggregate partition summaries may report non-identifying counts and proportions.

## Frozen final-holdout assignment rule

The final partition is fixed at **5,771 development patients and 1,924 final-holdout patients**. The latter is the nearest integer to 25% of 7,695. Assignment uses seed `20260829` and is created only after every R5/R6 rule preceding the split has been written.

The primary allocation cells are the Cartesian product of:

- drusen biological state: 00, 11, 10, or 01, preserving right/left order; and
- acquisition pair: Canon/Canon, Nikon/Nikon, or mixed.

No quality, age, sex, or confirmatory endpoint is added to the allocation cells. This avoids overconstraining sparse cells, particularly the eight mixed-camera pairs. Integer targets use a frozen two-stage Hamilton largest-remainder apportionment. First allocate the 1,924 holdout places across the three acquisition-pair categories in proportion to their cohort counts. Then, within each acquisition category, allocate its places across the observed drusen states in proportion to their within-category counts. At either stage, take all floor(quota) values and allocate remaining places by descending fractional remainder. Exact fractional ties are resolved by smaller source-cell count and then lexical category/state. This gives the scarce mixed category its proportional total of two holdout patients while preserving state directionality.

Within each stratum, patients are ordered by the ascending hexadecimal SHA-256 digest of the UTF-8 string `20260829|patient_id`; the first patients up to the stratum target enter the holdout. This hash is a deterministic assignment device, not a security or anonymization claim. The manifest is finally sorted by numeric patient ID.

Post-assignment balance is descriptive for confirmatory endpoint states, image-field pair status, focus pair status, acquisition pair, sex, and age category. It cannot trigger reseeding, rerandomization, or patient exchange. If an implementation fails the frozen cohort size, stratum totals, or one-partition-per-patient assertions, it stops instead of modifying the rule.

## Immutability

- Raw BRSET is read-only.
- No image or metadata row is deleted or rewritten.
- Eligibility is derived deterministically from v1.0.2 metadata.
- A source-metadata SHA-256 and eligible-patient-set SHA-256 are recorded with the partition.
- A future cohort change requires a protocol amendment and a new independent holdout; it cannot silently alter the frozen assignment.
