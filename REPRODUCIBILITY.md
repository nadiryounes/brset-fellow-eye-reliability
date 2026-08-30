# Reproducibility workflow

## 1. Configure local data

For all R7 stages, place BRSET at the repository-relative path
`data/BRSET_v1.0.2/`. A symlink to the credentialed release is acceptable, for
example:

```bash
mkdir -p data
ln -s /absolute/path/to/brazilian-ophthalmological/1.0.2 data/BRSET_v1.0.2
```

Alternatively, edit only the local path fields in
`configs/r7_frozen_config.json` to point to the credentialed release. Such a
path-only edit is operational and must not change any scientific parameter,
seed, partition hash, endpoint, or analysis setting.

The R0 audit and patient-partition generator separately support:

```bash
export BRSET_ROOT=/absolute/path/to/brazilian-ophthalmological/1.0.2
```

`BRSET_ROOT` is not read by the ordinary R7 configuration loader. Setting it
alone is therefore insufficient for R7 unless the repository-relative symlink
or the local config path is also present.

The frozen metadata SHA-256 is checked before partition generation.

## 2. Audit the release

```bash
python scripts/r0_brset_audit.py \
  --root "$BRSET_ROOT" \
  --out r0_audit \
  --hash-images
```

The audit output is local and git-ignored because several tables are row- or patient-level.

## 3. Generate the patient-level partition

```bash
BRSET_ROOT="$BRSET_ROOT" python scripts/generate_frozen_patient_partition.py
```

This writes `r6_frozen_split/` locally. The expected patient manifest SHA-256 is recorded in `configs/r7_frozen_config.json`. Do not publish the generated manifest.

If a non-default private output directory is necessary, set `BRSET_SPLIT_DIR` and update the local config path only. Such a path change is operational, not scientific.

## 4. Verify development isolation

```bash
python r7_run.py verify-split
python r7_run.py verify-development-files
```

The holdout firewall prevents holdout embeddings or predictions before the development-completion and release markers exist.

## 5. Development execution

```bash
python r7_run.py extract-development --batch-size 16
python r7_run.py run-development
python r7_run.py analyze-development
python r7_run.py validate-pre-holdout
```

Review the generated validation report. Scientific specifications must not be altered in response to development results.

## 6. Holdout execution

Only after the pre-holdout validation passes:

```bash
python r7_run.py release-holdout
python r7_run.py extract-holdout --batch-size 16
python r7_run.py run-holdout
python r7_run.py analyze-holdout
python r7_run.py finalize-provenance
```

The original study generated holdout predictions once. A reproduction run should preserve its own immutable provenance and must not use holdout results for model selection.

## 7. Statistical analysis

```bash
python scripts/run_r8_analysis.py
```

This script verifies the local R7 ledger before reading sealed predictions and writes aggregate R8 outputs. It does not train or recalibrate a model.

## 8. Tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/audit_public_release.py
```

## Public/private artifact boundary

Only source code, protocols, tests, configuration, and aggregate result tables belong in the public repository. Raw data, partitions, embeddings, predictions, fitted objects, and patient-level diagnostics remain private and git-ignored.

## Realized reliability-model environment

The sealed outputs were produced with scikit-learn 1.5.2. The Ridge calls used
the realized defaults `solver='auto'`, `tol=0.0001`, and `max_iter=None`.
Reproduction should use the locked environment in `requirements.txt` because
defaults may vary across library versions.

The development disease-head outer assessments and final holdout were
patient-isolated. Inner meta-selection used pooled out-of-fold logits; because
the disease heads producing training-row logits can include other assessment
folds in their training data, this introduces a second-order cross-fold
dependency in internal meta-selection. It cannot directly create final-holdout
leakage, but the inner meta-level is not strictly fully nested. A future
replication should use fully nested meta-level cross-fitting.
