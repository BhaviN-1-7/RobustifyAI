from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from pathlib import Path
import subprocess
import uuid
import os
import json
from typing import Dict, Optional, List

app = FastAPI()

# Increase multipart limits at FastAPI app level
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Create app with increased limits
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Override multipart parser limits
import starlette.formparsers
starlette.formparsers.MultiPartParser.max_files = 10000
starlette.formparsers.MultiPartParser.max_fields = 10000

ROOT = Path(__file__).resolve().parent.parent
ROBUSTIFY = ROOT / "RobustifyAI"
RUN_SCRIPT = Path(os.getenv("RUN_SCRIPT") or (ROBUSTIFY / "run_evaluation.py")).resolve()
PYTHON_BIN = Path(os.getenv("PYTHON_BIN") or (ROBUSTIFY / "venv" / "Scripts" / "python.exe")).resolve()
DEFAULT_REPORTS = Path(os.getenv("REPORTS_DIR") or (ROBUSTIFY / "reports")).resolve()

jobs: Dict[str, dict] = {}


class EvalRequest(BaseModel):
    dataset: str
    model: str
    seed: int = 1234
    out_dir: Optional[str] = None


EVAL_KEYWORDS = [
    ("Starting Evaluate()", 1, "Environment"),
    ("Split info:", 2, "Data split"),
    ("Model loaded successfully", 4, "Load Model"),
    ("Running inference", 6, "Model inference"),
    ("Wrote predictions.csv", 7, "Metrics"),
    ("Calibration", 12, "Calibration"),
    ("Wrote compact JSON", 16, "Summary"),
    ("Wrote HTML report", 17, "Generate Report"),
]


@app.post("/api/evaluate")
def evaluate(req: EvalRequest):
    job_id = str(uuid.uuid4())
    out_dir = Path(req.out_dir or (DEFAULT_REPORTS / job_id))
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PYTHON_BIN), str(RUN_SCRIPT),
        "--dataset", req.dataset,
        "--model", req.model,
        "--out", str(out_dir),
        "--seed", str(req.seed),
    ]

    log_path = out_dir / "job.log"
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        cmd, 
        cwd=str(ROBUSTIFY), 
        stdout=log_file, 
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}
    )
    jobs[job_id] = {"status": "running", "out_dir": str(out_dir), "pid": proc.pid, "log": str(log_path)}
    return {"job_id": job_id, "status": "running"}


def _estimate_progress(log_text: str) -> Dict[str, Optional[str]]:
    max_step = 0
    current = None
    for key, step_idx, label in EVAL_KEYWORDS:
        if key in log_text:
            if step_idx > max_step:
                max_step = step_idx
                current = label
    total = 17
    pct = int(round((max_step / total) * 100))
    return {"progress": pct, "current_step": current}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        # Attempt to recover job from filesystem if server restarted
        candidate_out = (DEFAULT_REPORTS / job_id)
        if candidate_out.exists() and candidate_out.is_dir():
            log_path = candidate_out / "job.log"
            jobs[job_id] = {
                "status": "running",
                "out_dir": str(candidate_out),
                "pid": None,
                "log": str(log_path) if log_path.exists() else None,
            }
            job = jobs[job_id]
        else:
            return JSONResponse({"error": "not found"}, status_code=404)
    out_dir = Path(job["out_dir"])            
    progress = {"progress": 0, "current_step": None}
    log_path = Path(job.get("log")) if job.get("log") else None
    if log_path and log_path.exists():
        try:
            txt = log_path.read_text(encoding="utf-8")
            progress = _estimate_progress(txt)
        except Exception:
            pass
    if (out_dir / "report.json").exists() and (out_dir / "report.html").exists():
        job["status"] = "completed"
    return {"job_id": job_id, "status": job["status"], "out_dir": job["out_dir"], **progress}


@app.get("/api/report/{job_id}.json")
def report_json(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
    path = Path(job["out_dir"]) / "report.json"
    if not path.exists():
        return JSONResponse({"error": "report.json not ready"}, status_code=404)
    with open(path, "r", encoding="utf-8") as f:
        return JSONResponse(json.load(f))


@app.get("/api/report/{job_id}.html")
def report_html(job_id: str, download: Optional[bool] = False):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
    path = Path(job["out_dir"]) / "report.html"
    if not path.exists():
        return JSONResponse({"error": "report.html not ready"}, status_code=404)
    headers = {"Content-Disposition": f"{'attachment' if download else 'inline'}; filename=report.html"}
    return FileResponse(path, headers=headers, media_type="text/html")


@app.get("/api/logs/{job_id}")
def get_logs(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
    log_val = job.get("log")
    if not log_val:
        return PlainTextResponse("", status_code=200)
    log_path = Path(log_val)
    if not log_path.exists():
        return PlainTextResponse("", status_code=200)
    try:
        content = log_path.read_text(encoding="utf-8")
    except Exception:
        content = ""
    return PlainTextResponse(content)


@app.post("/api/upload_evaluate_zip")
async def upload_and_evaluate_zip(
    model_file: UploadFile = File(...),
    dataset_zip: UploadFile = File(...),
    seed: int = Form(1234)
):
    """Simple ZIP-only upload endpoint that avoids multipart file limits"""
    try:
        job_id = str(uuid.uuid4())
        out_dir = (DEFAULT_REPORTS / job_id)
        uploads_dir = out_dir / "uploads"
        dataset_dir = out_dir / "dataset"
        model_dir = out_dir / "models"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model file
        model_path = model_dir / "uploaded_model.h5"
        with open(model_path, "wb") as f:
            content = await model_file.read()
            f.write(content)

        # Save and extract ZIP file
        ds_zip_path = uploads_dir / "dataset.zip"
        with open(ds_zip_path, "wb") as f:
            content = await dataset_zip.read()
            f.write(content)
        
        import zipfile
        with zipfile.ZipFile(ds_zip_path, 'r') as zf:
            zf.extractall(dataset_dir)

        # Launch evaluation
        cmd = [
            str(PYTHON_BIN), str(RUN_SCRIPT),
            "--dataset", str(dataset_dir),
            "--model", str(model_path),
            "--out", str(out_dir),
            "--seed", str(seed),
        ]
        log_path = out_dir / "job.log"
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROBUSTIFY),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )
        jobs[job_id] = {"status": "running", "out_dir": str(out_dir), "pid": proc.pid, "log": str(log_path)}
        return {"job_id": job_id, "status": "running"}
    
    except Exception as e:
        print(f"Error in upload_and_evaluate_zip: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Upload failed: {str(e)}"}, status_code=500)


@app.post("/api/upload_evaluate_folder")
async def upload_and_evaluate_folder(
    model_file: UploadFile = File(...),
    dataset_files: List[UploadFile] = File(...),
    seed: int = Form(1234)
):
    """Folder upload endpoint with higher file limits"""
    try:
        job_id = str(uuid.uuid4())
        out_dir = (DEFAULT_REPORTS / job_id)
        uploads_dir = out_dir / "uploads"
        dataset_dir = out_dir / "dataset"
        model_dir = out_dir / "models"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model file
        model_path = model_dir / "uploaded_model.h5"
        with open(model_path, "wb") as f:
            content = await model_file.read()
            f.write(content)

        # Save dataset files
        for file in dataset_files:
            if file.filename:
                # Preserve folder structure from webkitRelativePath
                file_path = dataset_dir / file.filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)

        # Launch evaluation
        cmd = [
            str(PYTHON_BIN), str(RUN_SCRIPT),
            "--dataset", str(dataset_dir),
            "--model", str(model_path),
            "--out", str(out_dir),
            "--seed", str(seed),
        ]
        log_path = out_dir / "job.log"
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROBUSTIFY),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )
        jobs[job_id] = {"status": "running", "out_dir": str(out_dir), "pid": proc.pid, "log": str(log_path)}
        return {"job_id": job_id, "status": "running"}
    
    except Exception as e:
        print(f"Error in upload_and_evaluate_folder: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Upload failed: {str(e)}"}, status_code=500)


@app.post("/api/upload_evaluate")
async def upload_and_evaluate(request: Request):
    try:
        print("Starting upload_and_evaluate...")
        form = await request.form()
        print(f"Form data keys: {list(form.keys())}")
        
        # Extract files and form data
        model_file = form.get("model_file")
        dataset_zip = form.get("dataset_zip") 
        dataset_files = form.getlist("dataset_files")
        
        # Handle seed conversion safely
        try:
            seed = int(form.get("seed", 1234))
        except (ValueError, TypeError):
            seed = 1234
        
        print(f"Received upload request - model_file: {getattr(model_file, 'filename', None) if model_file else None}")
        print(f"dataset_zip: {getattr(dataset_zip, 'filename', None) if dataset_zip else None}")
        print(f"dataset_files: {len(dataset_files) if dataset_files else 0} files")
        print(f"seed: {seed}")
        
        if not model_file or not hasattr(model_file, 'filename'):
            return JSONResponse({"error": "model_file is required"}, status_code=400)
        
        job_id = str(uuid.uuid4())
        out_dir = (DEFAULT_REPORTS / job_id)
        uploads_dir = out_dir / "uploads"
        dataset_dir = out_dir / "dataset"
        model_dir = out_dir / "models"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model file
        model_path = model_dir / "uploaded_model.h5"
        with open(model_path, "wb") as f:
            content = await model_file.read()
            f.write(content)

        # Handle dataset either as zip or as folder files
        if dataset_zip and hasattr(dataset_zip, 'filename'):
            ds_zip_path = uploads_dir / "dataset.zip"
            with open(ds_zip_path, "wb") as f:
                content = await dataset_zip.read()
                f.write(content)
            import zipfile
            with zipfile.ZipFile(ds_zip_path, 'r') as zf:
                zf.extractall(dataset_dir)
        elif dataset_files and len(dataset_files) > 0:
            for uf in dataset_files:
                if hasattr(uf, 'filename') and uf.filename:
                    rel = uf.filename.replace("..", "").lstrip("/\\")
                    dest = dataset_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with open(dest, "wb") as out:
                        content = await uf.read()
                        out.write(content)
        else:
            return JSONResponse({"error": "dataset not provided - please upload either a ZIP file or select a folder"}, status_code=400)

        # Launch evaluation
        cmd = [
            str(PYTHON_BIN), str(RUN_SCRIPT),
            "--dataset", str(dataset_dir),
            "--model", str(model_path),
            "--out", str(out_dir),
            "--seed", str(seed),
        ]
        log_path = out_dir / "job.log"
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROBUSTIFY),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )
        jobs[job_id] = {"status": "running", "out_dir": str(out_dir), "pid": proc.pid, "log": str(log_path)}
        return {"job_id": job_id, "status": "running"}
    
    except Exception as e:
        print(f"Error in upload_and_evaluate: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Upload failed: {str(e)}"}, status_code=500)


@app.get("/api/report/{job_id}.pdf")
def report_pdf(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
    html_path = Path(job["out_dir"]) / "report.html"
    if not html_path.exists():
        return JSONResponse({"error": "report.html not ready"}, status_code=404)
    try:
        import pdfkit  # requires wkhtmltopdf installed
        pdf_path = Path(job["out_dir"]) / "report.pdf"
        pdfkit.from_file(str(html_path), str(pdf_path))
        return FileResponse(pdf_path, media_type="application/pdf", filename="report.pdf")
    except Exception as e:
        return JSONResponse({"error": f"pdf generation not available: {e}"}, status_code=501)


