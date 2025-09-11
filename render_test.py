# render_test.py
import os, json, io, base64, traceback
from jinja2 import Environment, FileSystemLoader

# Build a safe sample report_data
report_data = {
    "seed": 1234,
    "steps": {
        "env": {"python": "3.8.10", "lib_versions": {"numpy":"1.24","tensorflow":"2.12"}},
        "data_list": {"classes": {"cats": 2, "dogs": 2}},
        "metrics": {"accuracy": 0.5, "precision": {"cats":0.5, "dogs":0.5}, "recall": {"cats":0.5,"dogs":0.5}},
        "robustness": {"gaussian_noise": 0.4},
        "efficiency": {"params": 123456, "model_size_mb": 1.2, "latency_ms": 12.3},
        "plots": {}  # empty but present
    }
}

try:
    # load template relative to this file
    env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates")))
    template = env.get_template("report_template.html")
    html = template.render(report=report_data)
    out = os.path.join("reports", "render_test.html")
    os.makedirs("reports", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("Rendered OK →", out)
except Exception as e:
    print("Render failed:")
    traceback.print_exc()
