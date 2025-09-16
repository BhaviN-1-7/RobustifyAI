export type StartEvalParams = {
  dataset: string;
  model: string;
  seed?: number;
  out_dir?: string;
};

export type StartEvalResponse = {
  job_id: string;
  status: string;
};

export type StatusResponse = {
  job_id: string;
  status: "running" | "completed" | "failed";
  out_dir: string;
  progress?: number;
  current_step?: string | null;
};

export async function startEvaluation(params: StartEvalParams): Promise<StartEvalResponse> {
  const res = await fetch("/api/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    throw new Error(`Failed to start evaluation: ${res.status}`);
  }
  return res.json();
}

export async function getStatus(jobId: string): Promise<StatusResponse> {
  const res = await fetch(`/api/status/${jobId}`);
  if (!res.ok) {
    throw new Error(`Failed to get status: ${res.status}`);
  }
  return res.json();
}

export async function fetchReport(jobId: string): Promise<any> {
  const res = await fetch(`/api/report/${jobId}.json`);
  if (!res.ok) {
    throw new Error("Report not ready");
  }
  return res.json();
}

export function reportHtmlUrl(jobId: string): string {
  return `/api/report/${jobId}.html`;
}

export async function fetchLogs(jobId: string): Promise<string> {
  const res = await fetch(`/api/logs/${jobId}`);
  if (!res.ok) return "";
  return res.text();
}

export async function uploadAndEvaluate(model: File, datasetZip: File, seed: number): Promise<StartEvalResponse> {
  const form = new FormData();
  form.append("model_file", model);
  form.append("dataset_zip", datasetZip);
  form.append("seed", String(seed));
  const res = await fetch("/api/upload_evaluate", { method: "POST", body: form });
  if (!res.ok) throw new Error("Failed to upload and start evaluation");
  return res.json();
}

export async function uploadAndEvaluateFolder(model: File, datasetFiles: FileList, seed: number): Promise<StartEvalResponse> {
  const form = new FormData();
  form.append("model_file", model);
  form.append("seed", String(seed));
  Array.from(datasetFiles).forEach((file) => {
    form.append("dataset_files", file, file.webkitRelativePath || file.name);
  });
  const res = await fetch("/api/upload_evaluate_folder", { method: "POST", body: form });
  if (!res.ok) throw new Error("Failed to upload folder and start evaluation");
  return res.json();
}

export function reportPdfUrl(jobId: string): string {
  return `/api/report/${jobId}.pdf`;
}


