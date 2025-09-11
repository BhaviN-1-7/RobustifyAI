# run_evaluation.py
import argparse, os, sys, traceback
from eval_core import Evaluate

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Path to dataset directory")
    p.add_argument("--model", required=True, help="Path to model.h5")
    p.add_argument("--out", default="reports", help="Output directory for report")
    p.add_argument("--seed", type=int, default=1234, help="Random seed")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    try:
        result = Evaluate(
            dataset_dir=args.dataset,
            model_path=args.model,
            out_dir=args.out,
            seed=args.seed
        )
        print("✅ Report generated at:", os.path.join(args.out, "report.html"))
        print("✅ JSON log at:", os.path.join(args.out, "report.json"))
    except Exception as e:
        tb = traceback.format_exc()
        with open(os.path.join(args.out,"report.html"),"w") as f:
            f.write(f"<html><body><h1>Evaluation failed</h1><pre>{tb}</pre></body></html>")
        sys.exit(1)
