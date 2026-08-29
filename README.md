# BRSET fellow-eye reliability analysis

Reproducible code and non-restricted aggregate outputs for a bilateral retinal-AI failure-analysis study using BRSET v1.0.2.

## Scientific question

The study asks whether output from the fellow eye provides incremental information for predicting own-eye model loss beyond a frozen own-eye probability-based reliability baseline. It does **not** test whether bilateral input improves disease classification, establish clinical utility, or provide external validation.

The primary endpoint is the BRSET retinal-specialist binary annotation for drusen presence versus absence. Increased cup-disc and diabetic retinopathy are fixed-sequence confirmatory endpoints.

## Data dependency and access

This repository does not include BRSET images, metadata rows, patient identifiers, split assignments, predictions, embeddings, or fitted models. BRSET v1.0.2 must be obtained from the official PhysioNet release:

- <https://physionet.org/content/brazilian-ophthalmological/1.0.2/>
- <https://doi.org/10.13026/mysn-8b26>

Access is governed by PhysioNet credentialing, required training, and the applicable data-use requirements. The BRSET dataset license and data-use terms are separate from this repository's software license.

See [DATA_ACCESS.md](DATA_ACCESS.md) before running any stage.

## Repository contents

- `src/`: frozen R7 preprocessing, embeddings, nested validation, calibration, reliability, firewall, inference, and provenance code.
- `scripts/r0_brset_audit.py`: dataset-integrity audit.
- `scripts/generate_frozen_patient_partition.py`: deterministic patient-level 75%/25% partition generator.
- `r7_run.py`: staged development and holdout execution interface.
- `scripts/run_r8_analysis.py`: analysis-only derivations from sealed R7 outputs.
- `configs/r7_frozen_config.json`: path-sanitized public copy of the frozen scientific configuration. Only local path fields differ from the execution copy.
- `protocol/`: prespecified endpoint, estimand, cohort, validation, inference, and narrative guardrails.
- `results/aggregate/`: non-restricted aggregate tables used for reporting.
- `tests/`: automated implementation tests.

No patient-level output is included.

## Environment

The frozen execution used Python 3.11 with the versions in `requirements.txt`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The frozen Torch/Torchvision versions may require a platform-compatible wheel source on systems other than the original Intel macOS environment.

## Reproduction stages

1. Obtain BRSET v1.0.2 from PhysioNet.
2. Place it locally at `data/BRSET_v1.0.2/`, or set `BRSET_ROOT`.
3. Run the R0 integrity audit.
4. Generate the deterministic patient partition locally. The patient-level manifest remains git-ignored.
5. Run development stages and the pre-holdout firewall.
6. Release and execute the holdout stage only after the development validation gate passes.
7. Run R8 analysis against the sealed local predictions.

Exact commands and expected boundaries are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Frozen patient-level protocol

- One right-eye and one left-eye image per eligible patient.
- 5,771 development patients and 1,924 untouched holdout patients.
- Five outer and three inner patient-level folds in development.
- Frozen ConvNeXt-Tiny ImageNet representation and L2 logistic head.
- Temperature/Platt calibration candidates selected inside development only.
- Own-eye NLL as the primary reliability target.
- Patient-averaged MSE contrast, `Delta_R = Risk(R0) - Risk(R1)`.
- Patient-cluster BCa bootstrap with 10,000 resamples.

The public repository does not expose the patient-to-partition mapping. Authorized BRSET users can regenerate it deterministically from the frozen script and verified metadata release.

## Regenerable outputs

With authorized local data, users can regenerate dataset audits, the frozen partition, embeddings, out-of-fold development probabilities, calibration and reliability fits, one-time holdout predictions, aggregate statistical tables, and figures. Generated patient-level artifacts remain local and are excluded by `.gitignore`.

The included aggregate tables are publication-supporting summaries, not substitutes for the credentialed dataset.

## Tests and release audit

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/audit_public_release.py
```

The model-API tests download the frozen Torchvision weight if it is not already cached. Data-dependent firewall and fold tests are skipped until an authorized local BRSET copy and locally generated patient partition are present. The release audit rejects common credentials, absolute workstation paths, raw-image formats, patient-level CSV columns, model artifacts, and forbidden directories.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Until the associated article has a final bibliographic record, cite the repository and BRSET dataset release separately.

## License

Original source code in this repository is released under the MIT License. This license does not apply to BRSET data, pretrained model weights, or third-party dependencies. Users must comply with their respective terms.

## Authors

- Younes Nadir — ORCID: <https://orcid.org/0000-0001-7493-3792>
- Khalid Boukhdir — ORCID: <https://orcid.org/0000-0002-4758-7086>
