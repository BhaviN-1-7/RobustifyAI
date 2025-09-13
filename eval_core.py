# eval_core.py
"""
Full feature evaluation core for the RobustifyAI hackathon.
Implements:
 - environment snapshot
 - dataset summary (supports train/test subfolders or single-folder stratified split)
 - preprocessing using common_preprocess.preprocess_fp
 - inference (handles sigmoid single-output or softmax multi-output)
 - per-class KPIs (precision/recall/f1/support), micro/macro
 - confusion matrix plot (base64)
 - ROC / PR curves (binary + multiclass)
 - bootstrap 95% CI (accuracy)
 - baseline (logistic regression on flattened pixels)
 - paired permutation test
 - calibration: Brier score, simple ECE (equal-width bins)
 - robustness tests (Gaussian noise, blur, JPEG, occlusion)
 - explainability placeholders (Grad-CAM/IG) — basic placeholder code included
 - efficiency metrics (params, model size, avg latency)
 - outputs: compact report.json (no images) + report.html (embedded images)
"""

import os, json, platform, pkg_resources, traceback, logging, time, copy, io, base64
from jinja2 import Environment, FileSystemLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from PIL import Image, ImageFilter, ImageOps

import warnings
warnings.filterwarnings("ignore")

# TF/Keras
from keras.models import load_model
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc, brier_score_loss
)
from sklearn.preprocessing import label_binarize

# numerical helpers
from scipy.special import expit
from scipy.ndimage import gaussian_filter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -------------------------
# Helpers: environment + dataset
# -------------------------
def get_environment_info():
    libs = {}
    for lib in ["numpy", "pandas", "scikit-learn", "tensorflow", "keras", "matplotlib", "seaborn"]:
        try:
            libs[lib] = pkg_resources.get_distribution(lib).version
        except Exception:
            libs[lib] = "not installed"
    return {"python": platform.python_version(), "lib_versions": libs}


def get_dataset_summary(dataset_dir):
    """
    Returns {"classes": {class_name: count, ...}}
    If dataset_dir does not exist returns {"classes": {}}
    """
    classes = {}
    if not os.path.isdir(dataset_dir):
        return {"classes": {}}
    for cls in sorted(os.listdir(dataset_dir)):
        p = os.path.join(dataset_dir, cls)
        if os.path.isdir(p):
            files = [f for f in os.listdir(p) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            classes[cls] = len(files)
    return {"classes": classes}


# -------------------------
# Image loading (uses user's common_preprocess)
# -------------------------
try:
    # user should provide this module
    from common_preprocess import preprocess_fp
except Exception as e:
    logging.warning("common_preprocess.preprocess_fp not found or failed to import. Falling back to internal loader.")
    from keras.utils import load_img, img_to_array
    def preprocess_fp(fp, target_size=(150,150)):
        img = load_img(fp, target_size=target_size)
        arr = img_to_array(img).astype("float32") / 255.0
        return arr


def load_images_from_files(filepaths, labels, target_size):
    X = [preprocess_fp(fp, target_size) for fp in filepaths]
    return np.array(X), np.array(labels)


# -------------------------
# Plot helpers (return base64 png strings)
# -------------------------
def plot_confusion_matrix_base64(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8"); plt.close(fig)
    return f"data:image/png;base64,{data}"


def plot_roc_pr_base64_and_scores(y_true, y_probs, class_names):
    """
    Compute per-class ROC AUC and PR AUC and return (roc_scores, pr_scores, roc_img_b64, pr_img_b64)
    roc_scores/pr_scores: {"per_class": {class: auc_or_None}, "macro": float_or_None}
    """
    roc_scores = {"per_class": {}, "macro": None}
    pr_scores = {"per_class": {}, "macro": None}
    roc_img_b64 = None
    pr_img_b64 = None

    try:
        y_true = np.asarray(y_true)
        y_probs = np.asarray(y_probs)
        n_classes = len(class_names)

        # binary case
        if n_classes == 2:
            # y_probs might be shape (N,) or (N,1) or (N,2)
            if y_probs.ndim == 1 or (y_probs.ndim == 2 and y_probs.shape[1] == 1):
                y_score = expit(y_probs).ravel()
            else:
                y_score = y_probs[:, 1].ravel()
            if len(np.unique(y_true)) >= 2:
                roc_auc = float(roc_auc_score((y_true==1).astype(int), y_score))
                roc_scores["per_class"][class_names[1]] = roc_auc
                roc_scores["macro"] = roc_auc
                prec, rec, _ = precision_recall_curve((y_true==1).astype(int), y_score)
                pr_auc = float(auc(rec, prec))
                pr_scores["per_class"][class_names[1]] = pr_auc
                pr_scores["macro"] = pr_auc

                # plot ROC
                fpr, tpr, _ = roc_curve((y_true==1).astype(int), y_score)
                fig, ax = plt.subplots(figsize=(6,4))
                ax.plot(fpr, tpr, label=f"{class_names[1]} (AUC={roc_auc:.3f})")
                ax.plot([0,1],[0,1],"k--",linewidth=0.7)
                ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC")
                ax.legend(loc="lower right", fontsize="small")
                buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); buf.seek(0)
                roc_img_b64 = base64.b64encode(buf.read()).decode("utf-8"); plt.close(fig)

                fig, ax = plt.subplots(figsize=(6,4))
                ax.plot(rec, prec, label=f"{class_names[1]} (AP={pr_auc:.3f})")
                ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall")
                ax.legend(loc="lower left", fontsize="small")
                buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); buf.seek(0)
                pr_img_b64 = base64.b64encode(buf.read()).decode("utf-8"); plt.close(fig)

        else:
            # multiclass: one-vs-rest
            # need binarized truth
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
            aucs = []
            pr_aucs = []
            fig1, ax1 = plt.subplots(figsize=(6,4))
            fig2, ax2 = plt.subplots(figsize=(6,4))
            for i, cls in enumerate(class_names):
                if np.unique(y_true_bin[:, i]).size < 2:
                    roc_scores["per_class"][cls] = None
                    pr_scores["per_class"][cls] = None
                    continue
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                auc_val = float(auc(fpr, tpr))
                aucs.append(auc_val)
                ax1.plot(fpr, tpr, label=f"{cls} (AUC={auc_val:.3f})")
                prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
                pr_auc_val = float(auc(rec, prec))
                pr_aucs.append(pr_auc_val)
                ax2.plot(rec, prec, label=f"{cls} (AP={pr_auc_val:.3f})")
                roc_scores["per_class"][cls] = auc_val
                pr_scores["per_class"][cls] = pr_auc_val
            if aucs:
                roc_scores["macro"] = float(np.mean(aucs))
            if pr_aucs:
                pr_scores["macro"] = float(np.mean(pr_aucs))

            ax1.plot([0,1],[0,1],"k--",linewidth=0.7)
            ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC Curves")
            ax1.legend(loc="lower right", fontsize="small")
            buf = io.BytesIO(); fig1.savefig(buf, format="png", bbox_inches="tight"); buf.seek(0)
            roc_img_b64 = base64.b64encode(buf.read()).decode("utf-8"); plt.close(fig1)

            ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("Precision-Recall Curves")
            ax2.legend(loc="lower left", fontsize="small")
            buf = io.BytesIO(); fig2.savefig(buf, format="png", bbox_inches="tight"); buf.seek(0)
            pr_img_b64 = base64.b64encode(buf.read()).decode("utf-8"); plt.close(fig2)
    except Exception as e:
        logging.warning("ROC/PR computation failed: %s", e)

    return roc_scores, pr_scores, (f"data:image/png;base64,{roc_img_b64}" if roc_img_b64 else None), (f"data:image/png;base64,{pr_img_b64}" if pr_img_b64 else None)


# -------------------------
# Baseline training (logistic regression on flattened pixels)
# -------------------------
def train_evaluate_baseline(X_train, y_train, X_test, y_test):
    try:
        if len(X_train) == 0:
            return {"accuracy": None, "precision": [], "recall": [], "f1": [], "y_pred": []}, None
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_test_flat = X_test.reshape(len(X_test), -1)
        clf = LogisticRegression(max_iter=400)
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": precision_score(y_test, y_pred, average=None, zero_division=0).tolist(),
            "recall": recall_score(y_test, y_pred, average=None, zero_division=0).tolist(),
            "f1": f1_score(y_test, y_pred, average=None, zero_division=0).tolist(),
            "y_pred": y_pred.tolist()
        }, clf
    except Exception as e:
        logging.warning("Baseline training failed: %s", e)
        return {"accuracy": None, "precision": [], "recall": [], "f1": [], "y_pred": []}, None


# -------------------------
# Bootstrap CI for accuracy
# -------------------------
def bootstrap_ci(y_true, y_pred, n_boot=1000, alpha=0.05, seed=1234):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n == 0:
        return {"lower": None, "upper": None}
    accs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        accs.append(accuracy_score(np.array(y_true)[idx], np.array(y_pred)[idx]))
    lower = float(np.percentile(accs, 100 * (alpha / 2)))
    upper = float(np.percentile(accs, 100 * (1 - alpha / 2)))
    return {"lower": lower, "upper": upper}


# -------------------------
# Paired permutation test for accuracy difference (model vs baseline preds)
# -------------------------
def paired_permutation_test_binary_correct(y_true, y_pred_model, y_pred_baseline, n_permutations=2000, seed=1234):
    try:
        y_true = np.array(y_true)
        y_pred_model = np.array(y_pred_model)
        y_pred_baseline = np.array(y_pred_baseline)
        if len(y_true) == 0 or len(y_pred_model) != len(y_true) or len(y_pred_baseline) != len(y_true):
            return {"observed_diff": None, "p_value": None}
        obs_diff = float(accuracy_score(y_true, y_pred_model) - accuracy_score(y_true, y_pred_baseline))
        rng = np.random.RandomState(seed)
        diffs = []
        for _ in range(n_permutations):
            mask = rng.rand(len(y_true)) < 0.5
            a = np.where(mask, y_pred_model, y_pred_baseline)
            b = np.where(mask, y_pred_baseline, y_pred_model)
            diffs.append(accuracy_score(y_true, a) - accuracy_score(y_true, b))
        p_val = float(np.mean(np.abs(diffs) >= abs(obs_diff)))
        return {"observed_diff": obs_diff, "p_value": p_val}
    except Exception as e:
        logging.warning("Permutation test failed: %s", e)
        return {"observed_diff": None, "p_value": None}


# -------------------------
# Calibration: Brier + ECE (equal-width bins)
# -------------------------
def calibration_metrics_and_plot(y_true, y_probs, n_bins=10):
    """
    y_probs: shape (N, n_classes) or (N,) for binary single-column
    returns: brier_score (overall), ece, calib_plot_b64 (image or None)
    """
    try:
        y_true = np.asarray(y_true)
        y_probs = np.asarray(y_probs)
        n = len(y_true)
        if n == 0:
            return None, None, None

        # For multiclass, compute Brier as mean squared error of one-hot - probs (sum across classes)
        if y_probs.ndim == 2 and y_probs.shape[1] > 1:
            # brier per sample = sum((y_onehot - probs)^2)
            y_onehot = np.zeros_like(y_probs)
            y_onehot[np.arange(n), y_true] = 1
            brier = float(np.mean(np.sum((y_onehot - y_probs) ** 2, axis=1)))
            # ECE: use max prob as confidence and predicted label
            confidences = np.max(y_probs, axis=1)
            preds = np.argmax(y_probs, axis=1)
            correctness = (preds == y_true).astype(int)
        else:
            # binary single-col or (N,1)
            if y_probs.ndim > 1 and y_probs.shape[1] == 1:
                probs = expit(y_probs).ravel()
            elif y_probs.ndim == 1:
                probs = expit(y_probs).ravel() if np.any(np.abs(y_probs) > 1) else y_probs.ravel()
            else:
                probs = y_probs.ravel()
            brier = float(brier_score_loss(y_true, probs))
            confidences = probs
            preds = (probs >= 0.5).astype(int)
            correctness = (preds == y_true).astype(int)

        # ECE
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        bin_counts = []
        bin_conf = []
        bin_acc = []
        for i in range(n_bins):
            lo = bins[i]; hi = bins[i+1]
            mask = (confidences >= lo) & (confidences < hi) if i < n_bins - 1 else (confidences >= lo) & (confidences <= hi)
            if np.sum(mask) == 0:
                bin_counts.append(0)
                bin_conf.append(None)
                bin_acc.append(None)
                continue
            bin_confidence = float(np.mean(confidences[mask]))
            bin_accuracy = float(np.mean(correctness[mask]))
            bin_counts.append(int(np.sum(mask)))
            bin_conf.append(bin_confidence)
            bin_acc.append(bin_accuracy)
            ece += (np.sum(mask) / n) * abs(bin_accuracy - bin_confidence)

        # calibration plot
        fig, ax = plt.subplots(figsize=(6,4))
        # plot perfect calibration line
        ax.plot([0,1],[0,1], "k--", linewidth=0.7)
        # scatter bin acc vs bin conf
        xs = [b for b in bin_conf if b is not None]
        ys = [a for a in bin_acc if a is not None]
        ax.scatter(xs, ys)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.set_title("Calibration plot")
        buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); buf.seek(0)
        calib_img_b64 = base64.b64encode(buf.read()).decode("utf-8"); plt.close(fig)
        return float(brier), float(ece), f"data:image/png;base64,{calib_img_b64}"
    except Exception as e:
        logging.warning("Calibration computation failed: %s", e)
        return None, None, None


# -------------------------
# Robustness corruptions (image-level utilities)
# -------------------------
def apply_gaussian_noise(img_arr, sigma=0.05):
    # img_arr assumed in [0,1]
    noisy = img_arr + np.random.normal(0, sigma, img_arr.shape)
    return np.clip(noisy, 0.0, 1.0)

def apply_blur(img_arr, radius=1):
    # convert to PIL and blur
    img = Image.fromarray((img_arr * 255).astype("uint8"))
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(img).astype("float32") / 255.0

def apply_jpeg_compression(img_arr, quality=30):
    img = Image.fromarray((img_arr * 255).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    img2 = Image.open(buf).convert("RGB")
    return np.asarray(img2).astype("float32") / 255.0

def apply_occlusion(img_arr, box_size_ratio=0.2):
    h, w = img_arr.shape[:2]
    box_h = int(h * box_size_ratio)
    box_w = int(w * box_size_ratio)
    top = np.random.randint(0, h - box_h + 1)
    left = np.random.randint(0, w - box_w + 1)
    img2 = img_arr.copy()
    img2[top:top+box_h, left:left+box_w, :] = 0.0
    return img2


# -------------------------
# Evaluate (main pipeline)
# -------------------------
def Evaluate(dataset_dir, model_path, out_dir, seed=1234, test_ratio=0.2):
    logging.info("Starting Evaluate()")
    logging.info(f"dataset_dir={dataset_dir}, model_path={model_path}, out_dir={out_dir}, seed={seed}")
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    env_info = get_environment_info()

    # detect train/test subfolders
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    use_external_split = os.path.isdir(train_dir) and os.path.isdir(test_dir)

    # data_list: always include train/test/full keys (empty if not available)
    data_list = {"train": {}, "test": {}, "full": {}}
    if use_external_split:
        data_list["train"] = get_dataset_summary(train_dir)
        data_list["test"] = get_dataset_summary(test_dir)
    else:
        data_list["full"] = get_dataset_summary(dataset_dir)

    # gather filepaths and labels
    def crawl_dir(base_dir):
        fps, labs = [], []
        classes = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        for idx, cls in enumerate(classes):
            cls_p = os.path.join(base_dir, cls)
            for fname in sorted(os.listdir(cls_p)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    fps.append(os.path.join(cls_p, fname))
                    labs.append(idx)
        return np.array(fps), np.array(labs), classes

    if use_external_split:
        X_train_files, y_train_labels, classes_train = crawl_dir(train_dir)
        X_test_files, y_test_labels, classes_test = crawl_dir(test_dir)
        classes = classes_test if classes_test else classes_train
        split_info = {"mode": "external_train_test", "n_train": int(len(y_train_labels)), "n_test": int(len(y_test_labels)), "seed": int(seed)}
    else:
        filepaths, labels, classes = crawl_dir(dataset_dir)
        if len(filepaths) == 0:
            raise ValueError("No images found under dataset_dir")
        X_train_files, X_test_files, y_train_labels, y_test_labels = train_test_split(
            filepaths, labels, test_size=test_ratio, stratify=labels, random_state=seed)
        split_info = {"mode": "stratified_split", "test_size": float(test_ratio), "seed": int(seed),
                      "n_train": int(len(y_train_labels)), "n_test": int(len(y_test_labels))}

    logging.info(f"Split info: {split_info}")

    # Load model
    try:
        model = load_model(model_path)
        logging.info("Model loaded successfully.")
    except Exception as e:
        tb = traceback.format_exc()
        logging.error("Model load failed: %s", e)
        return {"seed": seed, "steps": {"env": env_info, "data_list": data_list, "error": str(e), "traceback": tb}}

    # infer input size
    try:
        input_shape = model.input_shape[1:3]
        if input_shape[0] is None or input_shape[1] is None:
            input_shape = (150, 150)
    except Exception:
        input_shape = (150, 150)
    logging.info(f"Using target image size: {input_shape}")

    # load arrays
    X_train, y_train_arr = load_images_from_files(X_train_files, y_train_labels, input_shape)
    X_test, y_test_arr = load_images_from_files(X_test_files, y_test_labels, input_shape)
    logging.info(f"Loaded X_train={len(X_train)} X_test={len(X_test)}")

    # run inference (timed)
    logging.info(f"Running inference on {len(X_test)} images")
    start = time.time()
    y_probs = model.predict(X_test, verbose=0)
    elapsed = time.time() - start
    avg_time = elapsed / len(X_test) if len(X_test) > 0 else None
    logging.info(f"Inference time total={elapsed:.6f}s avg per image={avg_time:.6f}s")

    # normalize/transform y_probs to shape (N, n_classes)
    try:
        y_probs = np.asarray(y_probs)
        if y_probs.ndim == 1 or (y_probs.ndim == 2 and y_probs.shape[1] == 1):
            # expand single-col sigmoid to two columns (class 0 prob = 1-p)
            probs = expit(y_probs).ravel()
            y_probs = np.vstack([1 - probs, probs]).T
    except Exception as e:
        logging.warning("Post-process y_probs failed: %s", e)

    # predicted classes
    y_pred = np.argmax(y_probs, axis=1)

    n_classes = y_probs.shape[1]

    # save predictions CSV
    pred_df = pd.DataFrame({
        "file": [os.path.basename(p) for p in X_test_files],
        "true": [classes[int(i)] for i in y_test_arr],
        "pred": [classes[int(i)] for i in y_pred]
    })
    pred_csv = os.path.join(out_dir, "predictions.csv")
    pred_df.to_csv(pred_csv, index=False)
    logging.info(f"Wrote predictions.csv with {len(pred_df)} rows")

    # compute metrics
    acc = float(accuracy_score(y_test_arr, y_pred))
    prec_arr = precision_score(y_test_arr, y_pred, average=None, zero_division=0)
    rec_arr = recall_score(y_test_arr, y_pred, average=None, zero_division=0)
    f1_arr = f1_score(y_test_arr, y_pred, average=None, zero_division=0)
    precision = dict(zip(classes, [float(x) for x in prec_arr]))
    recall = dict(zip(classes, [float(x) for x in rec_arr]))
    f1 = dict(zip(classes, [float(x) for x in f1_arr]))

    # support counts
    support_counts = Counter(y_test_arr)
    support = {cls: int(support_counts[i]) for i, cls in enumerate(classes)}

    # macro / micro
    macro_prec = float(precision_score(y_test_arr, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_test_arr, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test_arr, y_pred, average="macro", zero_division=0))
    micro_prec = float(precision_score(y_test_arr, y_pred, average="micro", zero_division=0))
    micro_rec = float(recall_score(y_test_arr, y_pred, average="micro", zero_division=0))
    micro_f1 = float(f1_score(y_test_arr, y_pred, average="micro", zero_division=0))

    # bootstrap CI for accuracy
    ci = bootstrap_ci(y_test_arr, y_pred, n_boot=1000, alpha=0.05, seed=seed)

    # baseline training & evaluate
    baseline_metrics, baseline_clf = train_evaluate_baseline(X_train, y_train_arr, X_test, y_test_arr)

    # paired permutation test
    stats_test = {"observed_diff": None, "p_value": None}
    try:
        if baseline_metrics.get("y_pred"):
            stats_test = paired_permutation_test_binary_correct(y_test_arr, y_pred, baseline_metrics.get("y_pred", []))
    except Exception as e:
        logging.warning("Permutation test failed: %s", e)

    # confusion matrix plot
    try:
        cm_plot_b64 = plot_confusion_matrix_base64(y_test_arr, y_pred, classes)
    except Exception as e:
        logging.warning("Confusion matrix plotting failed: %s", e)
        cm_plot_b64 = None

    # ROC/PR plots and scores
    try:
        roc_scores, pr_scores, roc_plot_b64, pr_plot_b64 = plot_roc_pr_base64_and_scores(y_test_arr, y_probs, classes)
    except Exception as e:
        logging.warning("ROC/PR step failed: %s", e)
        roc_scores, pr_scores, roc_plot_b64, pr_plot_b64 = {"per_class": {}, "macro": None}, {"per_class": {}, "macro": None}, None, None

    # calibration metrics
    try:
        brier, ece, calib_plot_b64 = calibration_metrics_and_plot(y_test_arr, y_probs, n_bins=10)
    except Exception as e:
        logging.warning("Calibration step failed: %s", e)
        brier, ece, calib_plot_b64 = None, None, None

    # robustness tests (run small quick corruptions at one level)
    robustness_results = {}
    try:
        # apply corruptions on test set and evaluate accuracy drop (use small subset if large)
        n_robust = min(200, len(X_test))
        idxs = np.random.choice(len(X_test), size=n_robust, replace=False)
        clean_acc = accuracy_score(y_test_arr[idxs], np.argmax(y_probs[idxs], axis=1))
        # gaussian noise
        X_noisy = np.array([apply_gaussian_noise(X_test[i], sigma=0.05) for i in idxs])
        probs_noisy = model.predict(X_noisy, verbose=0)
        if probs_noisy.ndim == 1 or (probs_noisy.ndim == 2 and probs_noisy.shape[1] == 1):
            probs_noisy = np.vstack([1 - expit(probs_noisy).ravel(), expit(probs_noisy).ravel()]).T
        acc_noisy = accuracy_score(y_test_arr[idxs], np.argmax(probs_noisy, axis=1))
        # blur
        X_blur = np.array([apply_blur(X_test[i], radius=1) for i in idxs])
        probs_blur = model.predict(X_blur, verbose=0)
        if probs_blur.ndim == 1 or (probs_blur.ndim == 2 and probs_blur.shape[1] == 1):
            probs_blur = np.vstack([1 - expit(probs_blur).ravel(), expit(probs_blur).ravel()]).T
        acc_blur = accuracy_score(y_test_arr[idxs], np.argmax(probs_blur, axis=1))
        # jpeg
        X_jpeg = np.array([apply_jpeg_compression(X_test[i], quality=30) for i in idxs])
        probs_jpeg = model.predict(X_jpeg, verbose=0)
        if probs_jpeg.ndim == 1 or (probs_jpeg.ndim == 2 and probs_jpeg.shape[1] == 1):
            probs_jpeg = np.vstack([1 - expit(probs_jpeg).ravel(), expit(probs_jpeg).ravel()]).T
        acc_jpeg = accuracy_score(y_test_arr[idxs], np.argmax(probs_jpeg, axis=1))
        # occlusion
        X_occ = np.array([apply_occlusion(X_test[i], box_size_ratio=0.25) for i in idxs])
        probs_occ = model.predict(X_occ, verbose=0)
        if probs_occ.ndim == 1 or (probs_occ.ndim == 2 and probs_occ.shape[1] == 1):
            probs_occ = np.vstack([1 - expit(probs_occ).ravel(), expit(probs_occ).ravel()]).T
        acc_occ = accuracy_score(y_test_arr[idxs], np.argmax(probs_occ, axis=1))

        robustness_results = {
            "clean_acc_sample": float(clean_acc),
            "gaussian_noise_acc": float(acc_noisy),
            "blur_acc": float(acc_blur),
            "jpeg_acc": float(acc_jpeg),
            "occlusion_acc": float(acc_occ)
        }
    except Exception as e:
        logging.warning("Robustness tests failed or skipped: %s", e)
        robustness_results = {}

    # explainability placeholders
    explanation_results = {"gradcam": None, "integrated_gradients": None}
    # (Implementations of Grad-CAM and IG typically require model internals and will be added if requested.)

    # efficiency metrics
    try:
        params = int(getattr(model, "count_params", lambda: 0)())
    except Exception:
        params = None
    model_size_mb = round(os.path.getsize(model_path) / (1024 * 1024), 2) if os.path.exists(model_path) else None

    # assemble report_data
    report_data = {
        "seed": int(seed),
        "steps": {
            "env": env_info,
            "data_list": data_list,
            "split": split_info,
            "metrics": {
                "accuracy": acc,
                "accuracy_ci": ci,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "macro": {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1},
                "micro": {"precision": micro_prec, "recall": micro_rec, "f1": micro_f1}
            },
            "baseline": baseline_metrics,
            "stats_test": stats_test,
            "confusion_matrix": None,
            "roc": roc_scores,
            "pr": pr_scores,
            "calibration": {"brier": brier, "ece": ece},
            "robustness": robustness_results,
            "explainability": explanation_results,
            "efficiency": {
                "params": params,
                "model_size_mb": model_size_mb,
                "latency_s_per_image": float(avg_time) if avg_time is not None else None
            },
            "plots": {
                "confusion_matrix": cm_plot_b64,
                "roc": (roc_plot_b64 if roc_plot_b64 else None),
                "pr": (pr_plot_b64 if pr_plot_b64 else None),
                "calibration": (calib_plot_b64 if calib_plot_b64 else None),
                # explanation list would go here if available
                "explanations": []
            }
        }
    }

    # compact JSON (strip heavy base64 images)
    compact_report = copy.deepcopy(report_data)
    if "plots" in compact_report["steps"]:
        compact_report["steps"]["plots"] = {}
    json_path = os.path.join(out_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(compact_report, f, indent=2, ensure_ascii=False)
    logging.info("Wrote compact JSON to %s", json_path)

    # render HTML using templates/report_template.html (user should have template in templates/)
    try:
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("report_template.html")
        html_out = template.render(report=report_data)
        html_path = os.path.join(out_dir, "report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_out)
        logging.info("Wrote HTML report to %s", html_path)
    except Exception as e:
        tb = traceback.format_exc()
        err_html = f"<html><body><h1>Rendering failed</h1><pre>{tb}</pre></body></html>"
        with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
            f.write(err_html)
        logging.error("Rendering failed — wrote traceback to report.html")
        raise

    # Print minimal success lines for visibility
    print("✅ Wrote compact JSON to:", json_path)
    print("✅ Wrote HTML report to:", html_path)

    return report_data
