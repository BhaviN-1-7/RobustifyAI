# eval_core.py
import os
import json
import platform
import pkg_resources
import traceback
import time
import io
import base64
import logging

from jinja2 import Environment, FileSystemLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Use standalone keras import (works with your environment)
from keras.models import load_model

# sklearn imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------- Logging helper ----------
def setup_logger(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "debug.log")
    logger = logging.getLogger("eval_logger")
    logger.setLevel(logging.DEBUG)
    # Clear handlers if reloading
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_path

# ---------- Environment snapshot ----------
def get_environment_info(logger=None):
    libs = {}
    for lib in ["numpy", "pandas", "scikit-learn", "tensorflow", "keras", "matplotlib", "seaborn"]:
        try:
            libs[lib] = pkg_resources.get_distribution(lib).version
        except Exception:
            libs[lib] = "not installed"
    info = {"python": platform.python_version(), "lib_versions": libs}
    if logger: logger.debug(f"Environment info: {info}")
    return info

# ---------- Dataset helpers ----------
def get_dataset_summary(dataset_dir, logger=None):
    classes = {}
    if not os.path.isdir(dataset_dir):
        if logger: logger.warning(f"Dataset dir not found: {dataset_dir}")
        return {"classes": {}}
    for cls in sorted(os.listdir(dataset_dir)):
        p = os.path.join(dataset_dir, cls)
        if os.path.isdir(p):
            files = [f for f in os.listdir(p) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            classes[cls] = len(files)
    if logger: logger.debug(f"Dataset summary: {classes}")
    return {"classes": classes}

def make_file_list_and_labels(dataset_dir, logger=None):
    filepaths, labels = [], []
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    if logger: logger.debug(f"Discovered classes: {classes}")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for c in classes:
        p = os.path.join(dataset_dir, c)
        for fname in sorted(os.listdir(p)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                filepaths.append(os.path.join(p, fname))
                labels.append(class_to_idx[c])
    if logger: logger.debug(f"Found {len(filepaths)} images total")
    return filepaths, labels, classes

def stratified_split_files(filepaths, labels, test_size=0.2, seed=1234, logger=None):
    n_classes = len(set(labels))
    n_samples = len(labels)
    if logger: logger.debug(f"Attempting stratified split: n_samples={n_samples}, n_classes={n_classes}, test_size={test_size}")
    # If too small to ensure at least 1 sample per class in test, put everything in test
    if n_samples < 2 * n_classes or n_samples == 0:
        if logger: logger.warning("Dataset too small for stratified split — using all samples as test set.")
        return [], filepaths, [], labels
    # If test_size is so small that it would create <1 sample per class, increase it
    min_test = n_classes  # at least one sample per class
    test_count = max(1, int(round(test_size * n_samples)))
    if test_count < min_test:
        adjusted_test_size = min(0.5, float(min_test) / float(n_samples))
        if logger: logger.warning(f"Adjusting test_size from {test_size} to {adjusted_test_size} to ensure >=1 per class in test")
        test_size = adjusted_test_size
    X_train, X_test, y_train, y_test = train_test_split(
        filepaths, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    if logger: logger.debug(f"Split -> train:{len(X_train)} test:{len(X_test)}")
    return X_train, X_test, y_train, y_test

# ---------- Image loading ----------
def load_images_from_filelist(filepaths, target_size, logger=None):
    X = []
    good_files = []
    for p in filepaths:
        try:
            img = Image.open(p).convert("RGB")
            img = img.resize(target_size)
            X.append(np.array(img))
            good_files.append(p)
        except Exception as e:
            if logger: logger.warning(f"Failed to load image {p}: {e}")
    if len(X) == 0:
        return np.empty((0, *target_size, 3)), [], []
    X = np.array(X) / 255.0
    return X, good_files, list(range(len(good_files)))  # indices placeholder

# ---------- Plots (confusion matrix) ----------
def plot_confusion_matrix_base64(y_true, y_pred, class_names):
    try:
        cm = confusion_matrix(y_true, y_pred)
    except Exception:
        cm = None
    fig, ax = plt.subplots(figsize=(4, 4))
    if cm is None:
        ax.text(0.5, 0.5, "No confusion matrix", ha="center", va="center")
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{data}"

# ---------- Baseline training ----------
def train_evaluate_baseline(X_train_files, y_train, X_test_files, y_test, target_size, seed=1234, logger=None):
    if len(X_train_files) == 0 or len(X_test_files) == 0:
        if logger: logger.info("Skipping baseline training (no train or test samples).")
        return {"accuracy": None, "precision": [], "recall": [], "f1": [], "y_pred": []}, None, None
    Xtr, tr_files, _ = load_images_from_filelist(X_train_files, target_size, logger=logger)
    Xte, te_files, _ = load_images_from_filelist(X_test_files, target_size, logger=logger)
    if Xtr.size == 0 or Xte.size == 0:
        if logger: logger.warning("Baseline: after loading images, got empty arrays; skipping baseline.")
        return {"accuracy": None, "precision": [], "recall": [], "f1": [], "y_pred": []}, None, None
    n_features = Xtr.shape[1] * Xtr.shape[2] * Xtr.shape[3]
    Xtr_flat = Xtr.reshape((Xtr.shape[0], n_features))
    Xte_flat = Xte.reshape((Xte.shape[0], n_features))
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr_flat)
    Xte_s = scaler.transform(Xte_flat)
    clf = LogisticRegression(random_state=seed, max_iter=1000, solver="saga", multi_class="multinomial")
    clf.fit(Xtr_s, y_train)
    y_pred_baseline = clf.predict(Xte_s)
    acc = accuracy_score(y_test, y_pred_baseline)
    prec = precision_score(y_test, y_pred_baseline, average=None, zero_division=0)
    rec = recall_score(y_test, y_pred_baseline, average=None, zero_division=0)
    f1s = f1_score(y_test, y_pred_baseline, average=None, zero_division=0)
    metrics = {
        "accuracy": float(acc),
        "precision": [float(x) for x in prec],
        "recall": [float(x) for x in rec],
        "f1": [float(x) for x in f1s],
        "y_pred": y_pred_baseline.tolist()
    }
    if logger: logger.debug(f"Baseline metrics: {metrics}")
    return metrics, clf, scaler

# ---------- Paired permutation test ----------
def paired_permutation_test_binary_correct(y_true, y_pred_model, y_pred_baseline, n_permutations=1000, seed=1234, logger=None):
    if len(y_true) == 0 or len(y_pred_baseline) == 0:
        if logger: logger.debug("Skipping paired test (insufficient data).")
        return None, None
    rng = np.random.RandomState(seed)
    correct_model = (np.array(y_pred_model) == np.array(y_true)).astype(int)
    correct_base = (np.array(y_pred_baseline) == np.array(y_true)).astype(int)
    diffs = correct_model - correct_base
    observed = float(np.mean(diffs))
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([1, -1], size=len(diffs))
        perm = diffs * signs
        if abs(np.mean(perm)) >= abs(observed):
            count += 1
    p_value = (count + 1) / (n_permutations + 1)
    if logger: logger.debug(f"Permutation test observed={observed}, p={p_value}")
    return observed, p_value

# ---------- Safe helper to get model input size ----------
def infer_target_size_from_model(model, logger=None):
    # model.input_shape may look like (None, H, W, C) or (None, C, H, W)
    ishape = getattr(model, "input_shape", None)
    if logger: logger.debug(f"Model input_shape: {ishape}")
    if ishape is None:
        return (64, 64)
    # remove None dims and interpret
    try:
        if len(ishape) == 4:
            # typical formats
            _, d1, d2, d3 = ishape
            # assume channels last if d3 in {1,3}
            if d3 in (1, 3):
                return (int(d1 or 64), int(d2 or 64))
            # else assume channels first
            _, c, h, w = ishape
            return (int(h or 64), int(w or 64))
        elif len(ishape) == 3:
            _, h, w = ishape
            return (int(h or 64), int(w or 64))
    except Exception:
        if logger: logger.warning("Could not parse model.input_shape, defaulting to (64,64)")
    return (64, 64)

# ---------- Main Evaluate function ----------
def Evaluate(dataset_dir, model_path, out_dir, seed=1234):
    logger, log_path = setup_logger(out_dir)
    logger.info("Starting Evaluate()")
    logger.info(f"dataset_dir={dataset_dir}, model_path={model_path}, out_dir={out_dir}, seed={seed}")
    try:
        env_info = get_environment_info(logger=logger)
        data_info = get_dataset_summary(dataset_dir, logger=logger)

        # File list + split
        filepaths, labels, class_names = make_file_list_and_labels(dataset_dir, logger=logger)
        test_ratio = 0.2
        X_train_files, X_test_files, y_train, y_test = stratified_split_files(filepaths, labels, test_size=test_ratio, seed=seed, logger=logger)
        split_info = {"test_size": test_ratio, "seed": seed, "n_train": len(X_train_files), "n_test": len(X_test_files)}
        logger.info(f"Split info: {split_info}")

        # Load model
        try:
            model = load_model(model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load model")
            # Write a minimal report and exit
            report_data = {"seed": seed, "steps": {"env": env_info, "data_list": data_info, "error": f"Model could not be loaded: {e}"}}
            # save JSON
            with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            # render an error HTML
            template_dir = os.path.join(os.path.dirname(__file__), "templates")
            try:
                env = Environment(loader=FileSystemLoader(template_dir))
                # if template exists, render a simple error
                try:
                    template = env.get_template("report_template.html")
                    html_out = template.render(report=report_data)
                except Exception:
                    html_out = f"<html><body><h1>Model load failed</h1><pre>{traceback.format_exc()}</pre></body></html>"
                with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as fh:
                    fh.write(html_out)
            except Exception:
                with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as fh:
                    fh.write(f"<html><body><h1>Model load failed</h1><pre>{traceback.format_exc()}</pre></body></html>")
            return report_data

        # Infer target size
        target_size = infer_target_size_from_model(model, logger=logger)
        logger.info(f"Using target image size: {target_size}")

        # Load test images (note: if X_test_files is empty, try to use all files)
        if len(X_test_files) == 0 and len(filepaths) > 0:
            logger.info("No training set: using all files as test set.")
            X_test_files = filepaths
            y_test = labels

        X_test, good_test_files, _ = load_images_from_filelist(X_test_files, target_size, logger=logger)
        if X_test.size == 0:
            logger.warning("No test images could be loaded. Writing minimal report and exiting.")
            report_data = {
                "seed": seed,
                "steps": {
                    "env": env_info,
                    "data_list": data_info,
                    "split": split_info,
                    "error": "No test images loaded."
                }
            }
            with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            # render minimal html
            template_dir = os.path.join(os.path.dirname(__file__), "templates")
            try:
                env = Environment(loader=FileSystemLoader(template_dir))
                template = env.get_template("report_template.html")
                html_out = template.render(report=report_data)
                with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as fh:
                    fh.write(html_out)
            except Exception:
                with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as fh:
                    fh.write(f"<html><body><h1>No test images</h1></body></html>")
            return report_data

        # Run inference and measure latency
        logger.info(f"Running inference on {len(X_test)} images")
        start = time.perf_counter()
        y_probs = model.predict(X_test, verbose=0)
        end = time.perf_counter()
        total_time = end - start
        avg_time = total_time / len(X_test) if len(X_test) > 0 else None
        logger.info(f"Inference time total={total_time:.6f}s avg per image={avg_time:.6f}s")

        # Handle output shape: if model returns (N,) convert to 2-class probs
        y_probs = np.asarray(y_probs)
        if y_probs.ndim == 1:
            # binary probability of positive class -> convert
            y_probs = np.vstack([1 - y_probs, y_probs]).T
        if y_probs.ndim == 2:
            y_pred = np.argmax(y_probs, axis=1)
        else:
            # fallback
            y_pred = np.argmax(y_probs.reshape((y_probs.shape[0], -1)), axis=1)

        # Map y_test (which may be labels aligned to X_test_files)
        # Ensure lengths match: if we loaded good_test_files, reindex y_test accordingly
        if len(good_test_files) != 0 and len(good_test_files) != len(X_test_files):
            # create mapping from filepath -> label (from X_test_files and y_test)
            path_to_label = dict(zip(X_test_files, y_test))
            y_test_mapped = [path_to_label.get(p, None) for p in good_test_files]
            y_test_used = [int(v) for v in y_test_mapped if v is not None]
            # truncate predictions to same length if needed
            y_pred_used = y_pred[:len(y_test_used)]
        else:
            y_test_used = y_test
            y_pred_used = y_pred

        # Convert to numpy arrays
        y_test_used = np.array(y_test_used)
        y_pred_used = np.array(y_pred_used)

        # Save predictions CSV (keep small)
        try:
            df = pd.DataFrame({
                "file": good_test_files[: len(y_pred_used)],
                "true": [class_names[int(i)] for i in y_test_used[: len(y_pred_used)]],
                "pred": [class_names[int(i)] for i in y_pred_used[: len(y_pred_used)]]
            })
            df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
            logger.info(f"Wrote predictions.csv with {len(df)} rows")
        except Exception as e:
            logger.warning(f"Failed to write predictions.csv: {e}")

        # Primary metrics
        if len(y_test_used) > 0:
            accuracy = float(accuracy_score(y_test_used, y_pred_used))
            prec_arr = precision_score(y_test_used, y_pred_used, average=None, zero_division=0)
            rec_arr = recall_score(y_test_used, y_pred_used, average=None, zero_division=0)
            f1_arr = f1_score(y_test_used, y_pred_used, average=None, zero_division=0)
            precision = dict(zip(class_names, [float(x) for x in prec_arr]))
            recall = dict(zip(class_names, [float(x) for x in rec_arr]))
            f1 = dict(zip(class_names, [float(x) for x in f1_arr]))
            logger.info(f"Metrics: accuracy={accuracy}")
        else:
            accuracy = None
            precision = {}
            recall = {}
            f1 = {}

        # Confusion matrix plot (base64 string)
        cm_plot = plot_confusion_matrix_base64(y_test_used, y_pred_used, class_names) if len(y_test_used) > 0 else None

        # Baseline training
        baseline_metrics, baseline_clf, baseline_scaler = train_evaluate_baseline(
            X_train_files, y_train, X_test_files, y_test, target_size, seed=seed, logger=logger
        )

        # Paired permutation
        observed_diff, p_value = paired_permutation_test_binary_correct(y_test_used, y_pred_used, baseline_metrics.get("y_pred", []), n_permutations=1000, seed=seed, logger=logger)

        # Build report_data (plots kept for HTML but excluded from JSON)
        report_data = {
            "seed": seed,
            "steps": {
                "env": env_info,
                "data_list": data_info,
                "split": split_info,
                "metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                },
                "baseline": baseline_metrics,
                "stats_test": {"observed_diff": observed_diff, "p_value": p_value},
                "robustness": {},
                "efficiency": {
                    "params": int(getattr(model, "count_params", lambda: None)() or 0),
                    "model_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2) if os.path.exists(model_path) else None,
                    "latency_s_per_image": round(avg_time, 6) if avg_time is not None else None
                },
                "plots": {"confusion_matrix": cm_plot}
            }
        }

        # Save compact JSON (exclude plot binaries)
        json_path = os.path.join(out_dir, "report.json")
        report_data_for_json = json.loads(json.dumps(report_data))  # deep copy
        if "steps" in report_data_for_json and "plots" in report_data_for_json["steps"]:
            report_data_for_json["steps"]["plots"] = {}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data_for_json, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote compact JSON to {json_path}")

        # Render HTML (with embedded plots)
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("report_template.html")
        html_out = template.render(report=report_data)
        html_path = os.path.join(out_dir, "report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_out)
        logger.info(f"Wrote HTML report to {html_path}")

        return report_data

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Evaluation failed with an exception")
        # write error HTML + JSON
        err_report = {"seed": seed, "steps": {"env": {}, "error": str(e), "traceback": tb}}
        with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(err_report, f, indent=2, ensure_ascii=False)
        with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
            f.write(f"<html><body><h1>Evaluation failed</h1><pre>{tb}</pre></body></html>")
        return {"error": str(e), "traceback": tb}

