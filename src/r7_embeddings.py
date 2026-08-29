"""Deterministic frozen ConvNeXt-Tiny embedding extraction."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from .r7_config import project_path, sha256_file
from .r7_firewall import DEVELOPMENT, FINAL_HOLDOUT, authorize_stage
from .r7_preprocessing import preprocess_image


class AuthorizedImageDataset(Dataset):
    def __init__(self, records: list[dict[str, str]], config: dict[str, Any]):
        self.records = records
        self.config = config

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        path = Path(self.config["source"]["image_dir"]) / f"{record['image_id']}.jpg"
        return preprocess_image(path, self.config), index


class FrozenConvNeXtFeatures(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.final_norm = model.classifier[0]
        self.flatten = model.classifier[1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.features(inputs)
        values = self.avgpool(values)
        values = self.final_norm(values)
        return self.flatten(values)


def embedding_output_dir(config: dict[str, Any], stage: str) -> Path:
    return project_path(config, f"r7_artifacts/embeddings/{stage}")


def _ordered_identifier_records(
    config: dict[str, Any], stage: str
) -> list[dict[str, str]]:
    from .r7_firewall import stream_stage_metadata

    records = stream_stage_metadata(
        config,
        stage,
        columns=("image_id", "exam_eye"),
        purpose="embedding_extraction",
    )

    def key(record: dict[str, str]) -> tuple[int, int | str, int]:
        try:
            patient: tuple[int, int | str] = (0, int(record["patient_id"]))
        except ValueError:
            patient = (1, record["patient_id"])
        return (*patient, int(record["exam_eye"]))

    return sorted(records, key=key)


def build_frozen_encoder(config: dict[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    if config["model"]["weights_enum"] != "ConvNeXt_Tiny_Weights.IMAGENET1K_V1":
        raise ValueError("Unexpected frozen weight enum")
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    if weights.url != config["model"]["weight_url"]:
        raise ValueError(f"Torchvision weight URL changed: {weights.url}")
    model = convnext_tiny(weights=weights)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    encoder = FrozenConvNeXtFeatures(model).eval()
    with torch.inference_mode():
        probe = encoder(torch.zeros(1, 3, 224, 224))
    dimension = int(probe.shape[1])
    if probe.shape != (1, config["model"]["feature_dimension"]):
        raise ValueError(f"Unexpected feature shape: {tuple(probe.shape)}")
    provenance = {
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "weight_enum": str(weights),
        "weight_url": weights.url,
        "feature_dimension": dimension,
        "feature_definition": config["model"]["feature_definition"],
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in encoder.parameters()
        ),
        "evaluation_mode": not encoder.training,
    }
    return encoder, provenance


def extract_embeddings(
    config: dict[str, Any],
    stage: str,
    *,
    batch_size: int = 16,
    num_workers: int = 0,
) -> dict[str, Any]:
    authorize_stage(config, stage, purpose="embedding_extraction")
    output_dir = embedding_output_dir(config, stage)
    features_path = output_dir / "features.npy"
    index_path = output_dir / "embedding_index.csv"
    manifest_path = output_dir / "extraction_manifest.json"
    if any(path.exists() for path in (features_path, index_path, manifest_path)):
        raise RuntimeError(f"Refusing to overwrite {stage} embedding artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _ordered_identifier_records(config, stage)
    dataset = AuthorizedImageDataset(records, config)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    torch.manual_seed(config["cross_validation"]["training_seed_base"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    encoder, model_provenance = build_frozen_encoder(config)

    dimension = config["model"]["feature_dimension"]
    temporary_features = output_dir / "features.npy.tmp"
    matrix = np.lib.format.open_memmap(
        temporary_features,
        mode="w+",
        dtype=np.float32,
        shape=(len(records), dimension),
    )
    started = time.monotonic()
    seen: list[int] = []
    with torch.inference_mode():
        for tensors, indices in loader:
            values = encoder(tensors).detach().cpu().numpy().astype(np.float32)
            index_array = indices.numpy().astype(int)
            matrix[index_array] = values
            seen.extend(index_array.tolist())
    matrix.flush()
    del matrix
    if seen != list(range(len(records))):
        raise RuntimeError("Embedding extraction order changed or omitted an image")
    temporary_features.replace(features_path)

    temporary_index = output_dir / "embedding_index.csv.tmp"
    with temporary_index.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "image_id",
            "patient_id",
            "exam_eye",
            "partition",
            "feature_file",
            "feature_row",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for feature_row, record in enumerate(records):
            writer.writerow(
                {
                    "image_id": record["image_id"],
                    "patient_id": record["patient_id"],
                    "exam_eye": record["exam_eye"],
                    "partition": stage,
                    "feature_file": "features.npy",
                    "feature_row": feature_row,
                }
            )
    temporary_index.replace(index_path)

    manifest = {
        "stage": stage,
        "patients": len({record["patient_id"] for record in records}),
        "images": len(records),
        "feature_shape": [len(records), dimension],
        "feature_dtype": "float32",
        "feature_sha256": sha256_file(features_path),
        "index_sha256": sha256_file(index_path),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "platform": platform.platform(),
        "preprocessing": config["preprocessing"],
        "model": model_provenance,
        "labels_in_embedding_index": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_embeddings(
    config: dict[str, Any], stage: str
) -> tuple[np.ndarray, "Any"]:
    import pandas as pd

    authorize_stage(config, stage, purpose="embedding_read")
    output_dir = embedding_output_dir(config, stage)
    features_path = output_dir / "features.npy"
    index_path = output_dir / "embedding_index.csv"
    manifest_path = output_dir / "extraction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(features_path) != manifest["feature_sha256"]:
        raise RuntimeError("Embedding feature hash mismatch")
    if sha256_file(index_path) != manifest["index_sha256"]:
        raise RuntimeError("Embedding index hash mismatch")
    features = np.load(features_path, mmap_mode="r")
    index = pd.read_csv(index_path, dtype={"patient_id": str, "image_id": str})
    if tuple(features.shape) != tuple(manifest["feature_shape"]):
        raise RuntimeError("Embedding shape mismatch")
    return features, index
