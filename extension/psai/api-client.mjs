import {
  ApiError,
  isJobId,
  normalizeHealth,
  normalizeJob,
  normalizeJobs,
  parseApiResponse,
  validateBackendUrl,
  validateToken,
} from "./core.mjs";

const REQUEST_TIMEOUT_MS = 15_000;
const UPLOAD_TIMEOUT_MS = 45_000;

/** Authenticated API access to one validated, immutable loopback backend. */
export class PSAIApiClient {
  #baseUrl;
  #token;
  #fetch;

  constructor(settings, { fetchImpl = globalThis.fetch } = {}) {
    this.#baseUrl = validateBackendUrl(settings.baseUrl);
    this.#token = validateToken(settings.token);
    // Window.fetch rejects a client instance as its receiver in Chromium.
    this.#fetch = fetchImpl.bind(globalThis);
  }

  async health(options) {
    return normalizeHealth(await this.#request("/api/health", options));
  }

  async listJobs(options) {
    return normalizeJobs(await this.#request("/api/jobs", options));
  }

  async getJob(id, options) {
    if (!isJobId(id)) throw new ApiError("Mã tác vụ không hợp lệ.");
    const job = normalizeJob((await this.#request(`/api/jobs/${id}`, options)).job);
    if (job.id !== id) {
      throw new ApiError("Mã tác vụ trong phản hồi backend không khớp.");
    }
    return job;
  }

  async createJob(payload, { timeout = UPLOAD_TIMEOUT_MS } = {}) {
    const response = await this.#request("/api/jobs", {
      method: "POST",
      body: payload,
      timeout,
    });
    return normalizeJob(response.job);
  }

  async #request(path, { method = "GET", body, timeout = REQUEST_TIMEOUT_MS } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await this.#fetch(`${this.#baseUrl}${path}`, {
        method,
        headers: {
          Authorization: `Bearer ${this.#token}`,
          ...(body === undefined ? {} : { "Content-Type": "application/json" }),
        },
        body: body === undefined ? undefined : JSON.stringify(body),
        cache: "no-store",
        credentials: "omit",
        redirect: "error",
        signal: controller.signal,
      });
      return await parseApiResponse(response);
    } catch (error) {
      if (controller.signal.aborted) {
        throw new ApiError("Backend chưa phản hồi kịp. Kiểm tra cửa sổ backend; ứng dụng sẽ tự kết nối lại.");
      }
      if (error instanceof ApiError) throw error;
      throw new ApiError("Không liên lạc được với backend. Hãy kiểm tra backend đang chạy và địa chỉ cổng.");
    } finally {
      clearTimeout(timer);
    }
  }
}
