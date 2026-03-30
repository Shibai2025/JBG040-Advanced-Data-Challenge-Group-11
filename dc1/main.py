from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(step_name: str, cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 100)
    print(f"RUNNING STEP: {step_name}")
    print("COMMAND:", " ".join(cmd))
    print("=" * 100)

    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {step_name} (exit code {result.returncode})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full experiment pipeline from one entry point."
    )

    # Optional switches: still useful for debugging specific stages
    parser.add_argument("--run_optimizer", action="store_true", help="Run only optimizer experiment.")
    parser.add_argument("--run_imbalance", action="store_true", help="Run only imbalance experiment.")
    parser.add_argument("--run_threshold", action="store_true", help="Run only threshold experiment.")
    parser.add_argument("--run_evaluation", action="store_true", help="Run only final evaluation.")
    parser.add_argument("--run_gradcam", action="store_true", help="Run only Grad-CAM experiment.")
    parser.add_argument("--run_computer", action="store_true", help="Run only Computer Vision experiment.")

    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=25, help="Training batch size.")
    parser.add_argument("--val_batch_size", type=int, default=100, help="Validation/test batch size.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio.")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    base_dir = Path(__file__).resolve().parent
    python_exe = sys.executable

    # If user gives no stage flags, run the full pipeline by default
    any_stage_selected = any([
        args.run_optimizer,
        args.run_imbalance,
        args.run_threshold,
        args.run_evaluation,
        args.run_gradcam,
        args.run_computer,
    ])

    if any_stage_selected:
        run_optimizer = args.run_optimizer
        run_imbalance = args.run_imbalance
        run_threshold = args.run_threshold
        run_evaluation = args.run_evaluation
        run_gradcam = args.run_gradcam
        run_computer = args.run_computer
        print("Specific pipeline stages selected by user.")
    else:
        run_optimizer = True
        run_imbalance = True
        run_threshold = True
        run_evaluation = True
        run_gradcam = True
        run_computer = True
        print("No specific stage selected. Running the full pipeline by default.")

    common_flags: list[str] = []
    if args.force_cpu:
        common_flags.append("--force_cpu")

    try:
        if run_optimizer:
            cmd = [
                python_exe,
                "main_experiment_optimizer.py",
                "--nb_epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--val_batch_size",
                str(args.val_batch_size),
                "--seed",
                str(args.seed),
                "--val_ratio",
                str(args.val_ratio),
            ] + common_flags
            run_step("Optimizer Experiment", cmd, base_dir)

        if run_imbalance:
            cmd = [
                python_exe,
                "main_experiment_imbalance.py",
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--seed",
                str(args.seed),
                "--val_ratio",
                str(args.val_ratio),
            ] + common_flags
            run_step("Imbalance Experiment", cmd, base_dir)

        if run_threshold:
            cmd = [
                python_exe,
                "main_experiment_threshold.py",
                "--batch_size",
                str(args.val_batch_size),
                "--seed",
                str(args.seed),
                "--val_ratio",
                str(args.val_ratio),
            ] + common_flags
            run_step("Threshold Experiment", cmd, base_dir)

        if run_evaluation:
            cmd = [
                python_exe,
                "experiment_evaluation.py",
                "--batch_size",
                str(args.val_batch_size),
            ] + common_flags
            run_step("Experiment Evaluation", cmd, base_dir)

        if run_gradcam:
            cmd = [
                python_exe,
                "run_gradcam_experiment.py",
                "--comparison_mode",
                "same_sample_multi_model",
            ] + common_flags
            run_step("Grad-CAM Experiment", cmd, base_dir)

        if run_computer:
            cmd = [
                python_exe,
                "run_computer_vision_experiment.py",
            ]
            run_step("Computer Vision Image Cropping Experiment", cmd, base_dir)

        print("\n" + "=" * 100)
        print("Pipeline completed successfully.")
        print("All selected experiments and evaluations have finished.")
        print("=" * 100)

    except RuntimeError as exc:
        print("\n" + "=" * 100)
        print("Pipeline stopped.")
        print(str(exc))
        print("=" * 100)
        sys.exit(1)


if __name__ == "__main__":
    main()