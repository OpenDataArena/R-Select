#!/usr/bin/env python3
"""
Read all historical trial weights and loss from Optuna study.db.

Usage:
    # View all trials
    python read_optuna_trials.py --storage sqlite:////path/to/study.db --study_name your_study_name

    # Export to JSON
    python read_optuna_trials.py --storage sqlite:////path/to/study.db --study_name sft_weight_search --output path/to/trials.json

    python read_optuna_trials.py --storage sqlite:////path/to/study.db --study_name sft_weight_search --output path/to/trials.json

    # Export to CSV
    python read_optuna_trials.py --storage sqlite:////path/to/study.db --study_name your_study_name --output trials.csv

    # Show only the first 10
    python read_optuna_trials.py --storage sqlite:////path/to/study.db --study_name your_study_name --limit 10

    # Sort by loss
    python read_optuna_trials.py --storage sqlite:////path/to/study.db --study_name your_study_name --sort_by loss
"""

import argparse
import json
import optuna
import os
from typing import List, Dict, Any


def load_study(storage: str, study_name: str) -> optuna.Study:
    """Load an existing Optuna study"""
    return optuna.load_study(study_name=study_name, storage=storage)


def extract_trial_data(trial: optuna.trial.FrozenTrial) -> Dict[str, Any]:
    """Extract data from a single trial"""
    return {
        "trial_number": trial.number,
        "state": trial.state.name,
        "eval_loss": trial.value,
        "weights": trial.user_attrs.get("weights", {}),
        "is_warmup": trial.user_attrs.get("warmup", False),
        "params": trial.params,  # z parameters
        "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
        "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        "duration_seconds": (
            (trial.datetime_complete - trial.datetime_start).total_seconds()
            if trial.datetime_start and trial.datetime_complete else None
        ),
    }


def get_all_trials(study: optuna.Study) -> List[Dict[str, Any]]:
    """Get data from all trials"""
    return [extract_trial_data(t) for t in study.trials]


def print_trials_table(trials: List[Dict[str, Any]], top_n_weights: int = 5):
    """Print trials in table format"""
    print(f"\n{'='*100}")
    print(f"{'Trial':>6} | {'State':>10} | {'Loss':>10} | {'Duration':>10} | {'Top Weights'}")
    print(f"{'-'*100}")

    for t in trials:
        state = t["state"]
        loss = f"{t['eval_loss']:.6f}" if t["eval_loss"] is not None else "N/A"
        duration = f"{t['duration_seconds']:.1f}s" if t["duration_seconds"] else "N/A"

        # Get top N weight dimensions
        weights = t.get("weights", {})
        if weights:
            sorted_weights = sorted(weights.items(), key=lambda x: -x[1])[:top_n_weights]
            top_weights_str = ", ".join([f"{k}:{v:.3f}" for k, v in sorted_weights])
        else:
            top_weights_str = "N/A"

        warmup_marker = " [W]" if t.get("is_warmup") else ""
        print(f"{t['trial_number']:>6} | {state:>10} | {loss:>10} | {duration:>10} | {top_weights_str}{warmup_marker}")

    print(f"{'='*100}\n")


def ensure_output_dir_exists(output_path: str):
    """Ensure that the output file's parent directory exists."""
    dir_path = os.path.dirname(os.path.abspath(output_path))
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def export_to_json(trials: List[Dict[str, Any]], output_path: str):
    """Export trials to a JSON file"""
    ensure_output_dir_exists(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trials, f, ensure_ascii=False, indent=2)
    print(f"Exported {len(trials)} trials to {output_path}")


def export_to_csv(trials: List[Dict[str, Any]], output_path: str):
    """Export trials to a CSV file"""
    import csv

    if not trials:
        print("No trials to export")
        return

    ensure_output_dir_exists(output_path)

    # Collect all possible weight keys
    all_weight_keys = set()
    for t in trials:
        all_weight_keys.update(t.get("weights", {}).keys())
    all_weight_keys = sorted(all_weight_keys)

    # CSV columns
    fieldnames = [
        "trial_number", "state", "eval_loss", "is_warmup", "duration_seconds",
        "datetime_start", "datetime_complete"
    ] + [f"weight_{k}" for k in all_weight_keys]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for t in trials:
            row = {
                "trial_number": t["trial_number"],
                "state": t["state"],
                "eval_loss": t["eval_loss"],
                "is_warmup": t.get("is_warmup", False),
                "duration_seconds": t.get("duration_seconds"),
                "datetime_start": t.get("datetime_start"),
                "datetime_complete": t.get("datetime_complete"),
            }
            # Add each weight
            weights = t.get("weights", {})
            for k in all_weight_keys:
                row[f"weight_{k}"] = weights.get(k, 0.0)

            writer.writerow(row)

    print(f"Exported {len(trials)} trials to {output_path}")


def print_summary(study: optuna.Study, trials: List[Dict[str, Any]]):
    """Print summary statistics"""
    completed = [t for t in trials if t["state"] == "COMPLETE"]

    print("\n📊 Study Summary")
    print(f"{'='*50}")
    print(f"Study name: {study.study_name}")
    print(f"Total number of trials: {len(trials)}")
    print(f"Completed trials: {len(completed)}")
    print(f"Other statuses: {len(trials) - len(completed)}")

    if completed:
        losses = [t["eval_loss"] for t in completed if t["eval_loss"] is not None]
        if losses:
            print(f"\n📈 Loss statistics:")
            print(f"  Min loss: {min(losses):.6f}")
            print(f"  Max loss: {max(losses):.6f}")
            print(f"  Avg loss: {sum(losses)/len(losses):.6f}")

        # Best trial
        best = study.best_trial
        print(f"\n🏆 Best trial:")
        print(f"  Trial number: {best.number}")
        print(f"  Loss: {best.value:.6f}")
        print(f"  Weights:")
        weights = best.user_attrs.get("weights", {})
        sorted_weights = sorted(weights.items(), key=lambda x: -x[1])
        for k, v in sorted_weights[:10]:  # Only display top 10
            print(f"    {k}: {v:.4f}")
        if len(sorted_weights) > 10:
            print(f"    ... {len(sorted_weights) - 10} more dimensions")

    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Read all trials from an Optuna study")
    parser.add_argument("--storage", required=True, help="Optuna storage URL, e.g. sqlite:////path/to/study.db")
    parser.add_argument("--study_name", required=True, help="Study name")
    parser.add_argument("--output", "-o", default="", help="Output file path (.json or .csv)")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of displayed trials (0 means all)")
    parser.add_argument("--sort_by", choices=["number", "loss"], default="number", help="Sort type")
    parser.add_argument("--only_completed", action="store_true", help="Show only completed trials")
    parser.add_argument("--top_weights", type=int, default=5, help="Number of top weights to show in the table")
    args = parser.parse_args()

    # Load study
    print(f"Loading study: {args.study_name}")
    study = load_study(args.storage, args.study_name)

    # Get all trials
    trials = get_all_trials(study)

    # Filter
    if args.only_completed:
        trials = [t for t in trials if t["state"] == "COMPLETE"]

    # Sort
    if args.sort_by == "loss":
        trials = sorted(trials, key=lambda x: (x["eval_loss"] is None, x["eval_loss"] or float("inf")))
    else:
        trials = sorted(trials, key=lambda x: x["trial_number"])

    # Limit number
    display_trials = trials[:args.limit] if args.limit > 0 else trials

    # Print summary
    print_summary(study, trials)

    # Print table
    print(f"Showing {len(display_trials)}/{len(trials)} trials:")
    print_trials_table(display_trials, top_n_weights=args.top_weights)

    # Export
    if args.output:
        if args.output.endswith(".json"):
            export_to_json(trials, args.output)
        elif args.output.endswith(".csv"):
            export_to_csv(trials, args.output)
        else:
            print(f"Unsupported output format: {args.output}, please use .json or .csv")


if __name__ == "__main__":
    main()
