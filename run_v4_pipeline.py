"""Run the offline v3 -> v4 teacher/review/forecaster pipeline in order."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full v4 teacher-review release pipeline.")
    parser.add_argument("--skip-v3-build", action="store_true", help="reuse an existing behavior_windows_v3.parquet")
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    patch = os.getenv("TARGET_PATCH", "").strip()
    if not patch:
        print("TARGET_PATCH is required in .env", file=sys.stderr)
        return 1
    out_v3 = ROOT / f"output_v3/patch={patch}"
    out_v4 = ROOT / f"output_v4/patch={patch}"
    windows = out_v3 / "behavior_windows_v3.parquet"
    teacher_training = out_v4 / "timeline_teacher_training_v2.parquet"
    teacher_dir = out_v4 / "timeline_teacher_v2"
    teacher_scores = teacher_dir / "timeline_teacher_checkpoints_v2.parquet"
    review = out_v4 / "auto_review_windows_v2.parquet"
    models = out_v4 / "review_gated_models"

    if not args.skip_v3_build:
        run([sys.executable, "build_behavior_dataset.py"])
    elif not windows.exists():
        print(f"missing {windows}; cannot use --skip-v3-build", file=sys.stderr)
        return 2
    run([sys.executable, "build_timeline_teacher_dataset.py", "--v3-windows", str(windows), "--output", str(teacher_training)])
    run([sys.executable, "train_timeline_teacher.py", "--checkpoints", str(teacher_training), "--output-dir", str(teacher_dir)])
    run([sys.executable, "build_auto_review_dataset.py", "--windows", str(windows), "--teacher-checkpoints", str(teacher_scores), "--output", str(review)])
    run([sys.executable, "train_review_gated_forecaster.py", "--review-windows", str(review), "--output-dir", str(models), "--cv-folds", str(args.cv_folds)])
    print(f"v4 release complete: {out_v4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
