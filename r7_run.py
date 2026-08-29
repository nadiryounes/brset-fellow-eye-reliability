#!/usr/bin/env python3
"""Explicit stage CLI for the frozen R7 execution."""

from __future__ import annotations

import argparse
import json

from src.r7_analysis import run_development_analysis, run_holdout_analysis
from src.r7_config import load_config
from src.r7_development import run_development
from src.r7_embeddings import extract_embeddings
from src.r7_firewall import FINAL_HOLDOUT, verify_partition, verify_stage_image_files
from src.r7_holdout import execute_holdout_predictions
from src.r7_preholdout import release_holdout, validate_pre_holdout
from src.r7_provenance import finalize_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "verify-split",
            "verify-development-files",
            "extract-development",
            "run-development",
            "analyze-development",
            "validate-pre-holdout",
            "release-holdout",
            "extract-holdout",
            "run-holdout",
            "analyze-holdout",
            "finalize-provenance",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    config = load_config()

    if args.command == "verify-split":
        value = verify_partition(config)
        result = {
            "manifest_sha256": value.manifest_sha256,
            "development_patients": len(value.development_ids),
            "final_holdout_patients": len(value.holdout_ids),
        }
    elif args.command == "verify-development-files":
        result = verify_stage_image_files(config, "development")
    elif args.command == "extract-development":
        result = extract_embeddings(
            config, "development", batch_size=args.batch_size, num_workers=0
        )
    elif args.command == "run-development":
        result = run_development(config)
    elif args.command == "analyze-development":
        result = run_development_analysis(config)
    elif args.command == "validate-pre-holdout":
        result = validate_pre_holdout(config)
    elif args.command == "release-holdout":
        result = {"marker": str(release_holdout(config))}
    elif args.command == "extract-holdout":
        result = extract_embeddings(
            config, FINAL_HOLDOUT, batch_size=args.batch_size, num_workers=0
        )
    elif args.command == "run-holdout":
        result = execute_holdout_predictions(config)
    elif args.command == "analyze-holdout":
        result = run_holdout_analysis(config)
    elif args.command == "finalize-provenance":
        result = finalize_provenance(config)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
