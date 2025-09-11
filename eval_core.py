# eval_core.py
import os, json, platform, pkg_resources, traceback
from jinja2 import Environment, FileSystemLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import io, base64

import time


def get_environment_info():
    libs = {}
    for lib in ["numpy", "pandas", "scikit-learn", "tensorflow"]:
        try:
            libs[lib] = pkg_resources.get_distribution(lib).version
        except Exception:
            libs[lib] = "not installed"
    return {"python": platform.python_version(), "lib_versions": libs}

def get_dataset_summary(dataset_dir):
    classes = {}
    if not os.path.isdir(dataset_dir):
        return {"classes": {}}
    for cls in sorted(os.listdir(dataset_dir)):
        p = os.path.join(dataset_dir, cls)
        if os.path.isdir(p):
            files = [f for f in os.listdir(p) if f.lower().endswith((".jpg",".jpeg",".png"))]
            classes[cls] = len(files)
    return {"classes": classes}

def Evaluate(dataset_dir, model_path, out_dir, seed=1234):
    os.makedirs(out_dir, exist_ok=True)

    env_info = get_environment_info()
    data_info = get_dataset_summary(dataset_dir)

    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        return {
            "seed": seed,
            "steps": {
                "env": env_info,
                "data_list": data_info,
                "error": f"Model could not be loaded: {str(e)}"
            }
        }

    # Detect input size (H, W)
    input_shape = model.input_shape[1:3]
    X, y, class_names = load_images_from_dataset(dataset_dir, target_size=input_shape)

    # Run inference with latency measurement
    start = time.perf_counter()
    y_probs = model.predict(X, verbose=0)
    end = time.perf_counter()

    total_time = end - start
    avg_time_per_image = total_time / len(X) if len(X) > 0 else None
    y_pred = np.argmax(y_probs, axis=1)

    # Save predictions CSV
    df = pd.DataFrame({
        "true": [class_names[i] for i in y],
        "pred": [class_names[i] for i in y_pred]
    })
    df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    precision = dict(zip(class_names, precision_score(y, y_pred, average=None)))
    recall = dict(zip(class_names, recall_score(y, y_pred, average=None)))
    f1 = dict(zip(class_names, f1_score(y, y_pred, average=None)))

    # Confusion matrix plot
    cm_plot = plot_confusion_matrix(y, y_pred, class_names)

    # Report data
    report_data = {
        "seed": seed,
        "steps": {
            "env": env_info,
            "data_list": data_info,
            "metrics": {
                "accuracy": float(accuracy),
                "precision": precision,
                "recall": recall,
                "f1": f1
            },
            "robustness": {},
            "efficiency": {
                "params": model.count_params(),
                "model_size_mb": round(os.path.getsize(model_path) / (1024*1024), 2),
                "latency_s_per_image": round(avg_time_per_image, 6) if avg_time_per_image else None
            },
            "plots": {"confusion_matrix": cm_plot}
        }
    }

    # Save JSON
    json_path = os.path.join(out_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # Render HTML
    env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates")))
    template = env.get_template("report_template.html")
    html_out = template.render(report=report_data)
    html_path = os.path.join(out_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    return report_data

def load_images_from_dataset(dataset_dir, target_size):
    X, y, class_names = [], [], []
    classes = sorted(os.listdir(dataset_dir))
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(dataset_dir, cls)
        if os.path.isdir(cls_path):
            class_names.append(cls)
            for fname in os.listdir(cls_path):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(cls_path, fname)
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(target_size)
                    X.append(np.array(img))
                    y.append(idx)
    X = np.array(X) / 255.0  # normalize
    y = np.array(y)
    return X, y, class_names


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{data}"
