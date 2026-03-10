"use client";

import { useCallback, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type JobStatus = "pending" | "processing" | "completed" | "failed" | "cancelled";

interface JobState {
  id: string;
  status: JobStatus;
  progress_percent: number;
  current_step: string | null;
  error_message: string | null;
  output_path: string | null;
  video_filename: string | null;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [startTime, setStartTime] = useState("15");
  const [endTime, setEndTime] = useState("30");
  const [job, setJob] = useState<JobState | null>(null);
  const [uploading, setUploading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const pollStatus = useCallback(async (jobId: string) => {
    const res = await fetch(`${API_BASE}/api/v1/status/${jobId}`);
    if (!res.ok) throw new Error("Failed to fetch status");
    const data = await res.json();
    setJob({
      id: data.id,
      status: data.status,
      progress_percent: data.progress_percent ?? 0,
      current_step: data.current_step ?? null,
      error_message: data.error_message ?? null,
      output_path: data.output_path ?? null,
      video_filename: data.video_filename ?? null,
    });
    if (data.status === "processing" || data.status === "pending") {
      setTimeout(() => pollStatus(jobId), 1500);
    }
  }, []);

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setUploading(true);
    setJob(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("start_time", startTime);
      form.append("end_time", endTime);
      const res = await fetch(`${API_BASE}/api/v1/upload-video`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const err = await res.text();
        throw new Error(err || "Upload failed");
      }
      const { job_id } = await res.json();
      await pollStatus(job_id);
    } catch (err) {
      setJob({
        id: "",
        status: "failed",
        progress_percent: 0,
        current_step: null,
        error_message: err instanceof Error ? err.message : "Unknown error",
        output_path: null,
        video_filename: null,
      });
    } finally {
      setUploading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(URL.createObjectURL(f));
    }
  };

  const downloadUrl = job?.status === "completed" && job.id
    ? `${API_BASE}/api/v1/download/${job.id}`
    : null;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 font-sans">
      <div className="max-w-3xl mx-auto px-6 py-12">
        <header className="mb-12">
          <h1 className="text-3xl font-bold tracking-tight text-white">
            AI Video Dubbing
          </h1>
          <p className="mt-2 text-zinc-400">
            English → Hindi with voice cloning and lip sync
          </p>
        </header>

        <form onSubmit={handleUpload} className="space-y-6">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-6">
            <label className="block text-sm font-medium text-zinc-300 mb-2">
              Upload video
            </label>
            <label className="flex flex-col items-center justify-center w-full h-40 border-2 border-dashed border-zinc-700 rounded-lg cursor-pointer hover:border-amber-500/50 hover:bg-zinc-800/30 transition-colors">
              <input
                type="file"
                accept=".mp4,.avi,.mov,.mkv,.webm"
                onChange={handleFileChange}
                className="hidden"
              />
              {file ? (
                <span className="text-amber-400 font-medium">{file.name}</span>
              ) : (
                <span className="text-zinc-500">
                  Drop a video or click to browse
                </span>
              )}
              <span className="mt-1 text-xs text-zinc-500">
                MP4, AVI, MOV, MKV, WebM
              </span>
            </label>
          </div>

          <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-6">
            <label className="block text-sm font-medium text-zinc-300 mb-3">
              Segment (seconds)
            </label>
            <div className="flex gap-4 items-center">
              <div>
                <span className="text-xs text-zinc-500">Start</span>
                <input
                  type="number"
                  min={0}
                  step={0.5}
                  value={startTime}
                  onChange={(e) => setStartTime(e.target.value)}
                  className="mt-1 block w-24 rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-white focus:border-amber-500 focus:ring-1 focus:ring-amber-500"
                />
              </div>
              <span className="text-zinc-500 pt-5">→</span>
              <div>
                <span className="text-xs text-zinc-500">End</span>
                <input
                  type="number"
                  min={0}
                  step={0.5}
                  value={endTime}
                  onChange={(e) => setEndTime(e.target.value)}
                  className="mt-1 block w-24 rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-white focus:border-amber-500 focus:ring-1 focus:ring-amber-500"
                />
              </div>
              <span className="text-zinc-500 text-sm pt-5">
                e.g. 00:15–00:30
              </span>
            </div>
          </div>

          <button
            type="submit"
            disabled={!file || uploading}
            className="w-full rounded-lg bg-amber-500 px-6 py-3 font-medium text-zinc-900 hover:bg-amber-400 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {uploading ? "Uploading & processing…" : "Upload and process"}
          </button>
        </form>

        {job && (
          <div className="mt-8 rounded-xl border border-zinc-800 bg-zinc-900/50 p-6">
            <h2 className="text-lg font-semibold mb-4">Status</h2>
            <div className="space-y-4">
              <div className="flex justify-between text-sm">
                <span className="text-zinc-400">Status</span>
                <span
                  className={
                    job.status === "completed"
                      ? "text-emerald-400"
                      : job.status === "failed"
                        ? "text-red-400"
                        : "text-amber-400"
                  }
                >
                  {job.status}
                </span>
              </div>
              {(job.status === "processing" || job.status === "pending") && (
                <>
                  <div className="h-2 rounded-full bg-zinc-800 overflow-hidden">
                    <div
                      className="h-full bg-amber-500 transition-all duration-300"
                      style={{ width: `${job.progress_percent}%` }}
                    />
                  </div>
                  {job.current_step && (
                    <p className="text-sm text-zinc-500">{job.current_step}</p>
                  )}
                </>
              )}
              {job.status === "failed" && job.error_message && (
                <p className="text-sm text-red-400">{job.error_message}</p>
              )}
              {job.status === "completed" && downloadUrl && (
                <a
                  href={downloadUrl}
                  download={`dubbed_${job.video_filename || "video"}.mp4`}
                  className="inline-flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 font-medium text-white hover:bg-emerald-500"
                >
                  Download dubbed video
                </a>
              )}
            </div>
          </div>
        )}

        {previewUrl && file && (
          <div className="mt-8 rounded-xl border border-zinc-800 bg-zinc-900/50 p-6">
            <h2 className="text-lg font-semibold mb-4">Preview</h2>
            <video
              src={previewUrl}
              controls
              className="w-full rounded-lg"
            />
          </div>
        )}
      </div>
    </div>
  );
}
