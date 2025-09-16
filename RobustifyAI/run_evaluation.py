# run_evaluation.py
import argparse, os, sys, traceback
from eval_core import Evaluate

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run model evaluation and write report.html & report.json")
    p.add_argument("--dataset", required=True,
                   help="Path to dataset folder. If it has train/ and test/ subfolders, they are used. Otherwise dataset_dir should contain class subfolders and --test_ratio will be used.")
    p.add_argument("--model", required=True, help="Path to model.h5")
    p.add_argument("--out", default="reports", help="Output directory for report")
    p.add_argument("--seed", type=int, default=1234, help="Random seed")
    p.add_argument("--test_ratio", type=float, default=0.2,
                   help="Fraction to use as test split if no train/test subfolders exist")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    try:
        result = Evaluate(
            dataset_dir=args.dataset,
            model_path=args.model,
            out_dir=args.out,
            seed=args.seed,
            test_ratio=args.test_ratio
        )
        print("✅ Report generated at:", os.path.join(args.out, "report.html"))
        print("✅ JSON log at:", os.path.join(args.out, "report.json"))
    except Exception:
        tb = traceback.format_exc()
        with open(os.path.join(args.out, "report.html"), "w", encoding="utf-8") as f:
            f.write(f"<html><body><h1>Evaluation failed</h1><pre>{tb}</pre></body></html>")
        print("❌ Evaluation failed. Traceback written to report.html")
        sys.exit(1)
