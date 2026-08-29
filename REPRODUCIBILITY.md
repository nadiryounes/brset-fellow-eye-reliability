# Reproducibility workflow

## 1. Configure local data

Either place BRSET at `data/BRSET_v1.0.2/` or export:

```bash
export BRSET_ROOT=/absolute/path/to/brazilian-ophthalmological/1.0.2
```

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
