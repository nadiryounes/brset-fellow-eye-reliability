# R5 Endpoint Rubric Verification

Freeze date: 2026-08-29
Dataset release: BRSET v1.0.2
Purpose: verify operational endpoint meanings before prediction generation

## Sources examined

| Source | Status and evidence used |
|---|---|
| [Official PhysioNet BRSET v1.0.2 page](https://physionet.org/content/brazilian-ophthalmological/1.0.2/) | Current release documentation. It states that all images were labeled by a retinal-specialist ophthalmologist, lists pathology labels, defines every binary classification field as 1 present/0 absent, and records the v1.0.2 rename from `drusens` to `drusen`. |
| Local v1.0.2 release | `[LOCAL BRSET ROOT]/`; contains `label_brset.csv`, images, release checksums, license, and an index. The metadata SHA-256 already verified in R0 is `4cd000ec1f651deb5cfb4f018ed1dec7a874da59a20c47f1e59a421d4a112adf`. No local README or data dictionary is present. |
| [Original BRSET paper](https://doi.org/10.1371/journal.pdig.0000454) | States that a retinal specialist labeled the images using research-group criteria. It lists drusen among pathological classifications, defines the vertical cup-to-disc threshold used for the optic-disc anatomical classification, and specifies ICDR/SDRG grades and quality criteria. |
| [Official S1 data dictionary, `pdig.0000454.s001.docx`](https://journals.plos.org/digitalhealth/article/file?type=supplementary&id=info:doi/10.1371/journal.pdig.0000454.s001) | Downloaded from the PLOS supplementary-file endpoint solely for rubric inspection. SHA-256: `333ac96de199a747e0e7a25594b9904cc7fd25a075b80b7e1bfad76a058fdce1`. It defines `drusens`, `increased_cup_disc`, and `diabetic_retinopathy` only as 1 present/0 absent, and provides ICDR/SDRG grade definitions. |
| [Official BRSET GitHub repository](https://github.com/luisnakayama/BRSET) | Refers users to PhysioNet for dataset structure and contains no more detailed drusen rubric in its repository documentation. |
| R0 audit outputs | Establish v1.0.2 observed counts, completeness, laterality mapping, quality anomalies, and bilateral state structure without altering the release. |

The source search did not identify an official BRSET rule for drusen size, type, location, count, AMD grade, or AREDS category.

## Frozen endpoint definitions

### Primary endpoint: `drusen`

**Frozen operational definition:**

> BRSET retinal-specialist binary annotation for presence versus absence of drusen.

Encoding in v1.0.2 is `drusen=1` present and `drusen=0` absent. The original paper and S1 dictionary use the plural field name `drusens`; the v1.0.2 release note explicitly changes that spelling to `drusen` without documenting a criterion change.

No threshold or morphology rubric is supplied. Therefore this project will not infer or assert:

- hard versus soft drusen;
- drusen diameter, area, number, or location;
- macular versus peripheral drusen rules;
- AREDS category;
- AMD stage;
- that `drusen` and the separate BRSET `amd` label are interchangeable.

This limitation is scientifically material: 00, 11, 10, and 01 describe the supplied specialist annotations, not standardized drusen burden or AMD severity.

### Confirmatory endpoint: `increased_cup_disc`

**Frozen operational definition:**

> BRSET retinal-specialist binary annotation for presence versus absence of increased cup-to-disc ratio.

The current data dictionary defines only 1 present/0 absent. The original BRSET paper additionally states that the optic-disc anatomical classification used a vertical cup-to-disc ratio threshold of at least 0.65. This threshold is contextual source evidence; the project will not recompute it because no continuous cup-to-disc measurement is supplied in v1.0.2.

The label is a finding. It is not a glaucoma diagnosis and does not establish glaucomatous optic neuropathy, intraocular-pressure status, or visual-field loss.

### Confirmatory endpoint: `diabetic_retinopathy`

**Frozen operational definition:**

> BRSET retinal-specialist binary annotation for presence versus absence of diabetic retinopathy.

The release also provides ICDR and SDRG grades 0–4. Official documentation defines the grade levels, but it does not state that the binary label was mechanically derived as grade greater than zero.

R5 aggregate verification of v1.0.2 confirms that the relationship is not perfectly deterministic:

- ICDR: 14 binary-negative images have grade above 0, and 1 binary-positive image has grade 0;
- SDRG: 14 binary-negative images have grade above 0, and 6 binary-positive images have grade 0.

Accordingly, the binary field remains the confirmatory endpoint. ICDR/SDRG are secondary ordinal sensitivity outcomes; neither will be used to overwrite the binary label or adjudicate individual records.

## Quality-variable rubrics relevant to the endpoints

The original paper provides more detail than the binary metadata dictionary:

- `image_field` is adequate when the disc, macular center, and temporal arcades meet the paper’s disc-diameter visibility criteria;
- `focus` is adequate when third-generation branches can be identified within one disc diameter around the macula.

The metadata encodes 1 normal/adequate and 2 abnormal/inadequate. Two v1.0.2 records contain invalid `focus=0`; R0 identified them and they remain unevaluable, not abnormal.

These are human retrospective annotations. They are not assumed to be generated automatically at deployment.

## Label-provenance limitations

- The sources state that a retinal specialist labeled images but do not report endpoint-specific intergrader or intragrader reliability.
- No adjudication process or drusen morphology rubric is published in the examined sources.
- The original paper reports earlier-release counts that differ slightly from audited v1.0.2 counts. All analyses use v1.0.2 metadata and R0 counts.
- Binary labels are global image-level annotations, not lesion segmentations or quantitative burdens.

## Endpoint decision

The rubric gate is **resolved with a transparent limitation**. `drusen` remains the primary endpoint under the exact operational definition above. No additional clinical meaning will be imputed.

Frozen clean-pair states from R0 are:

| Endpoint | 00 | 11 | 10 | 01 | Total pairs |
|---|---:|---:|---:|---:|---:|
| `drusen` | 6,002 | 984 | 363 | 346 | 7,695 |
| `increased_cup_disc` | 5,829 | 1,171 | 399 | 296 | 7,695 |
| `diabetic_retinopathy` | 7,153 | 440 | 60 | 42 | 7,695 |

These four states remain distinct in all primary or confirmatory bilateral analyses.
