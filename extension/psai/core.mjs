export const DEFAULT_BACKEND = "http://127.0.0.1:8765";
export const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
export const ACTIVE_STATUSES = new Set(["queued", "running"]);
const STATUSES = new Set(["queued", "running", "completed", "failed"]);
const MODES = new Set(["full", "fast"]);
const DEVICES = new Set(["auto", "cpu", "cuda"]);
const OCR_OPTIONS = new Set(["auto", "none", "always"]);
const ID_PATTERN = /^[A-Za-z0-9_-]{1,128}$/;

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function validateBackendUrl(value) {
  const raw = String(value ?? "").trim();
  // Require a literal loopback address and an explicit port. URL alone normalizes
  // alternate host spellings and paths, which must not expand this allowlist.
  const match = /^http:\/\/127\.0\.0\.1:([0-9]{1,5})\/?$/.exec(raw);
  if (!match || Number(match[1]) < 1 || Number(match[1]) > 65535) {
    throw new Error("Địa chỉ phải có dạng http://127.0.0.1:8765, với cổng từ 1 đến 65535.");
  }
  return `http://127.0.0.1:${Number(match[1])}`;
}

export function validateToken(value) {
  const token = String(value ?? "").trim();
  if (!token || !/^[\x21-\x7E]{16,512}$/.test(token)) {
    throw new Error("Dán mã ghép nối do backend PSAI hiển thị (ít nhất 16 ký tự, không có khoảng trắng).");
  }
  return token;
}

export function validateFile(file, limit = MAX_UPLOAD_BYTES) {
  if (!file) throw new Error("Chọn một tài liệu trước khi bắt đầu.");
  if (!/\.(pdf|md|txt)$/i.test(file.name ?? "")) {
    throw new Error("Chỉ hỗ trợ tài liệu PDF, Markdown (.md) và văn bản (.txt).");
  }
  if (!Number.isSafeInteger(file.size) || file.size <= 0) {
    throw new Error("Tài liệu đang trống hoặc không đọc được.");
  }
  if (file.size > Math.min(limit, MAX_UPLOAD_BYTES)) {
    throw new Error(`Tài liệu vượt giới hạn ${formatBytes(Math.min(limit, MAX_UPLOAD_BYTES))}.`);
  }
  return file;
}

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1).replace(/\.0$/, "")} MB`;
}

export function normalizeHealth(payload) {
  if (payload?.status !== "ok" || typeof payload.version !== "string" ||
      !Number.isSafeInteger(payload.max_upload_bytes) || payload.max_upload_bytes < 1) {
    throw new ApiError("Backend trả về thông tin kết nối không hợp lệ.");
  }
  return { version: payload.version, maxUploadBytes: Math.min(payload.max_upload_bytes, MAX_UPLOAD_BYTES) };
}

export function normalizeJob(payload) {
  if (!payload || typeof payload.id !== "string" || !ID_PATTERN.test(payload.id) || typeof payload.filename !== "string" ||
      !STATUSES.has(payload.status) || !MODES.has(payload.mode) ||
      !DEVICES.has(payload.device) || !OCR_OPTIONS.has(payload.ocr)) {
    throw new ApiError("Backend trả về tác vụ không hợp lệ. Hãy kiểm tra phiên bản backend.");
  }
  return {
    id: payload.id,
    filename: payload.filename.slice(0, 500),
    status: payload.status,
    mode: payload.mode,
    device: payload.device,
    ocr: payload.ocr,
    stage: typeof payload.stage === "string" ? payload.stage.slice(0, 500) : "",
    created_at: typeof payload.created_at === "string" ? payload.created_at : "",
    updated_at: typeof payload.updated_at === "string" ? payload.updated_at : "",
    error: typeof payload.error === "string" ? payload.error.slice(0, 10000) : "",
    log_tail: Array.isArray(payload.log_tail)
      ? payload.log_tail.filter((line) => typeof line === "string").join("\n").slice(-24000)
      : typeof payload.log_tail === "string" ? payload.log_tail.slice(-24000) : "",
    results: payload.status === "completed" && payload.results && typeof payload.results === "object"
      ? payload.results : {},
  };
}

export function normalizeJobs(payload) {
  if (!Array.isArray(payload?.jobs)) throw new ApiError("Không đọc được lịch sử tác vụ từ backend.");
  return payload.jobs.map(normalizeJob).sort((a, b) => b.created_at.localeCompare(a.created_at));
}

export function resultUrl(baseUrl, path, jobId) {
  const base = validateBackendUrl(baseUrl);
  if (typeof path !== "string" || typeof jobId !== "string" || !ID_PATTERN.test(jobId)) return null;
  const match = /^\/results\/([A-Za-z0-9_-]{1,128})\/([A-Za-z0-9_-]+)\/([A-Za-z0-9][A-Za-z0-9._-]*)$/.exec(path);
  if (!match || match[1] !== jobId || match[3] === "." || match[3] === "..") return null;
  return new URL(path, base).href;
}

export function nextSelectedId(jobs, currentId) {
  if (jobs.some((job) => job.id === currentId)) return currentId;
  return jobs.find((job) => ACTIVE_STATUSES.has(job.status))?.id ?? jobs[0]?.id ?? null;
}

export async function parseApiResponse(response) {
  if (response.status === 401 || response.status === 403) {
    throw new ApiError("Mã ghép nối không đúng hoặc đã thay đổi. Dán mã từ backend rồi kết nối lại.", response.status);
  }
  let body;
  try {
    body = await response.json();
  } catch {
    throw new ApiError(response.ok
      ? "Backend trả về dữ liệu không phải JSON."
      : `Backend trả về lỗi HTTP ${response.status}.`, response.status);
  }
  if (!response.ok) {
    const detail = typeof body?.error === "string" ? body.error
      : typeof body?.error?.message === "string" ? body.error.message
      : typeof body?.detail === "string" ? body.detail : `Lỗi HTTP ${response.status}`;
    throw new ApiError(detail.slice(0, 2000), response.status);
  }
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    throw new ApiError("Backend trả về dữ liệu không hợp lệ.", response.status);
  }
  return body;
}
