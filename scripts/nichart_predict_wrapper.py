#!/usr/bin/env python3
"""NiChart-compatible inference wrapper for AmyloidPETNet.

This wrapper accepts a flat directory of NIfTI files, generates the CSV format
expected by predict.py, runs prediction, and copies the final CSV to a
user-specified output directory.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


NIFTI_SUFFIXES = (".nii", ".nii.gz")


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in NIFTI_SUFFIXES)


def _collect_nifti_files(input_dir: Path) -> list[str]:
    # NiChart usually provides a flat directory; scan one level and sort for deterministic ordering.
    files = [str(path.resolve()) for path in input_dir.iterdir() if path.is_file() and _is_nifti(path)]
    files.sort()
    return files


def _run_predict(
    predict_script: Path,
    model_dir: Path,
    dataset_csv: Path,
    cache_dir: Path,
    vis_dir: Path | None,
) -> None:
    # Call the original project entrypoint instead of importing it, since predict.py parses CLI args at import time.
    cmd = [
        sys.executable,
        str(predict_script),
        "--odir",
        str(model_dir),
        "--dataset",
        str(dataset_csv),
        "--cdir",
        str(cache_dir),
    ]
    if vis_dir is not None:
        cmd.extend(["--vdir", str(vis_dir)])

    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="NiChart wrapper for AmyloidPETNet prediction")
    parser.add_argument("--input-dir", required=True, help="Directory containing input .nii/.nii.gz files")
    parser.add_argument("--output-dir", required=True, help="Directory where output CSV will be written")
    parser.add_argument("--model-dir", default="/app/model", help="Directory containing model.pth")
    parser.add_argument("--cache-dir", default="/tmp", help="Writable cache directory")
    parser.add_argument("--vis-dir", default=None, help="Optional visualization output directory")
    parser.add_argument(
        "--output-csv-name",
        default="predictions.csv",
        help="Output CSV filename written inside --output-dir",
    )
    parser.add_argument(
        "--predict-script",
        default="/app/predict.py",
        help="Path to predict.py inside the container",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    cache_dir = Path(args.cache_dir)
    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    predict_script = Path(args.predict_script)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    if not predict_script.is_file():
        raise FileNotFoundError(f"predict script not found: {predict_script}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = _collect_nifti_files(input_dir)
    if not nifti_files:
        raise RuntimeError(
            f"no NIfTI files found in {input_dir}; expected .nii or .nii.gz files in a flat directory"
        )

    print(f"found {len(nifti_files)} NIfTI files")

    # Build the CSV schema expected by predict.py: a single column named "img_path".
    with tempfile.TemporaryDirectory(dir=str(cache_dir)) as temp_dir:
        temp_csv = Path(temp_dir) / "nichart_input.csv"
        pd.DataFrame({"img_path": nifti_files}).to_csv(temp_csv, index=False)

        _run_predict(
            predict_script=predict_script,
            model_dir=model_dir,
            dataset_csv=temp_csv,
            cache_dir=cache_dir,
            vis_dir=vis_dir,
        )

        # predict.py writes output as <odir>/<basename(dataset_csv)>, so expect the same CSV name under model_dir.
        model_csv = model_dir / temp_csv.name
        if not model_csv.is_file():
            raise FileNotFoundError(f"prediction output CSV not found at expected path: {model_csv}")

        # Copy into NiChart's designated output mount so results survive container teardown.
        final_csv = output_dir / args.output_csv_name
        shutil.copy2(model_csv, final_csv)
        print(f"wrote predictions: {final_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
