# BRSET data access and non-redistribution boundary

BRSET v1.0.2 is available through PhysioNet at <https://physionet.org/content/brazilian-ophthalmological/1.0.2/> (DOI: <https://doi.org/10.13026/mysn-8b26>).

Users must obtain access directly from PhysioNet and comply with its credentialing, training, license, and data-use requirements. This repository does not grant rights to BRSET and does not redistribute:

- retinal images;
- the source metadata table;
- patient or image identifiers;
- the patient-level frozen partition;
- patient-level predictions, losses, or reliability outputs;
- embeddings or fitted model objects.

The default local layout is:

```text
data/BRSET_v1.0.2/
├── label_brset.csv
└── fundus_photos/
```

The entire `data/` directory and all patient-level output directories are git-ignored. Do not weaken these exclusions when forking the repository.
