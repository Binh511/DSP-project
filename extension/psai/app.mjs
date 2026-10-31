import {
  ACTIVE_STATUSES, ApiError, DEFAULT_BACKEND, MAX_UPLOAD_BYTES,
  formatBytes, nextSelectedId, normalizeHealth, normalizeJob, normalizeJobs,
  parseApiResponse, resultUrl, validateBackendUrl, validateFile, validateToken,
} from "./core.mjs";

const $ = (id) => document.getElementById(id);
const STATUS_LABELS = { queued: "Đang chờ", running: "Đang xử lý", completed: "Hoàn tất", failed: "Có lỗi" };
const MODE_LABELS = { full: "Đầy đủ · AI cục bộ", fast: "Chạy nhanh · Không model học" };
const STAGE_LABELS = {
  queued: "Tài liệu đã tiếp nhận, đang chờ bộ xử lý.",
  source: "Đang đọc tài liệu và chuẩn bị văn bản nguồn…",
  extract: "Đang phân loại và trích xuất thông tin từ tài liệu…",
  retrieve: "Đang truy xuất, đối chiếu các đoạn nguồn liên quan…",
  plan: "Đang tổng hợp báo cáo và lập kế hoạch dự án…",
  completed: "Phân tích hoàn tất. Mở kết quả để đối chiếu nội dung và nguồn.",
  failed: "Tác vụ chưa hoàn tất. Xem thông báo và nhật ký bên dưới.",
};
const state = {
  settings: { baseUrl: DEFAULT_BACKEND, token: "" },
  connected: false, jobs: [], selectedId: null, file: null,
  submitting: false, uncertain: false, maxUploadBytes: MAX_UPLOAD_BYTES,
};
let generation = 0;
let pollTimer = null;
let activeRefresh = null;
let lastConnectionMessage = "";
let resultLinks = {};

function notice(message, error = false) {
  $("notice-text").textContent = message;
  $("notice").classList.toggle("error", error);
  $("notice").hidden = false;
}

function renderControls() {
  $("analyze-button").disabled = !state.connected || !state.file || state.submitting || state.uncertain;
  $("analyze-label").textContent = state.submitting ? "Đang gửi tài liệu…"
    : state.uncertain ? "Đang xác minh tác vụ…" : "Bắt đầu phân tích";
  $("refresh-jobs").disabled = !state.connected || state.submitting;
  $("document-file").disabled = state.submitting;
  $("remove-file").disabled = state.submitting;
  $("connect-button").disabled = state.submitting;
  $("backend-url").disabled = state.submitting;
  $("pairing-token").disabled = state.submitting;
}

function connectionStatus(phase, message) {
  state.connected = phase === "online";
  const labels = { offline: "Chưa kết nối", connecting: "Đang kết nối", online: "Đã kết nối", error: "Mất kết nối", auth: "Kiểm tra mã" };
  $("connection-dot").className = `status-dot ${phase === "auth" ? "error" : phase}`;
  $("connection-badge").className = `badge ${phase === "offline" ? "neutral" : phase === "auth" ? "error" : phase}`;
  $("connection-badge").textContent = labels[phase];
  if (message !== lastConnectionMessage) {
    $("connection-message").textContent = message;
    lastConnectionMessage = message;
  }
  renderControls();
}

function toggleConnection(open) {
  $("connection-form").hidden = !open;
  $("connection-toggle").setAttribute("aria-expanded", String(open));
  $("connection-toggle").textContent = open ? "Thu gọn ↑" : "Cài đặt kết nối ↓";
}

function api(path, { method = "GET", body, timeout = 15000 } = {}) {
  const settings = { ...state.settings };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  return fetch(`${settings.baseUrl}${path}`, {
    method,
    headers: { Authorization: `Bearer ${settings.token}`, ...(body ? { "Content-Type": "application/json" } : {}) },
    body: body ? JSON.stringify(body) : undefined,
    cache: "no-store", credentials: "omit", redirect: "error", signal: controller.signal,
  }).then(parseApiResponse).catch((error) => {
    if (error instanceof ApiError) throw error;
    throw new ApiError(error.name === "AbortError"
      ? "Backend chưa phản hồi kịp. Kiểm tra cửa sổ backend; ứng dụng sẽ tự kết nối lại."
      : "Không liên lạc được với backend. Hãy kiểm tra backend đang chạy và địa chỉ cổng.");
  }).finally(() => clearTimeout(timer));
}

function schedulePoll(epoch, delay) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(() => {
    if (epoch !== generation) return;
    if (!state.connected) connect(false);
    else refreshJobs(epoch).then(() => schedulePoll(epoch, state.jobs.some((job) => ACTIVE_STATUSES.has(job.status)) ? 2000 : 10000))
      .catch((error) => handleConnectionError(error, epoch));
  }, delay);
}

function handleConnectionError(error, epoch) {
  if (epoch !== generation) return;
  if (error.status === 401 || error.status === 403) {
    connectionStatus("auth", error.message);
    toggleConnection(true);
    clearTimeout(pollTimer);
    return;
  }
  connectionStatus("error", `${error.message} Tự thử lại sau 5 giây.`);
  schedulePoll(epoch, 5000);
}

async function persistSettings() {
  try {
    await chrome.storage.local.set({ psaiSettings: state.settings });
  } catch {
    notice("Không lưu được cài đặt vào trình duyệt. Bạn có thể cần nhập lại mã khi mở ứng dụng.");
  }
}

async function connect(save = true) {
  let settings;
  try {
    settings = save ? {
      baseUrl: validateBackendUrl($("backend-url").value),
      token: validateToken($("pairing-token").value),
    } : state.settings;
  } catch (error) {
    notice(error.message, true);
    return;
  }
  const epoch = ++generation;
  clearTimeout(pollTimer);
  if (settings.baseUrl !== state.settings.baseUrl || settings.token !== state.settings.token) {
    state.jobs = [];
    state.selectedId = null;
    state.uncertain = false;
    renderJobs();
  }
  state.settings = settings;
  $("backend-url").value = settings.baseUrl;
  connectionStatus("connecting", "Đang kiểm tra kết nối với backend PSAI…");
  $("connect-button").textContent = "Đang kết nối…";
  $("connect-button").disabled = true;
  if (save) await persistSettings();
  try {
    const health = normalizeHealth(await api("/api/health"));
    if (epoch !== generation) return;
    state.maxUploadBytes = health.maxUploadBytes;
    $("file-help").textContent = `PDF, Markdown, TXT · Tối đa ${formatBytes(health.maxUploadBytes)}`;
    await refreshJobs(epoch);
    if (epoch !== generation) return;
    connectionStatus("online", `Backend PSAI ${health.version} đang hoạt động tại ${settings.baseUrl}.`);
    renderJobs();
    toggleConnection(false);
    if (state.file) {
      try { validateFile(state.file, state.maxUploadBytes); }
      catch (error) { clearFile(); notice(error.message, true); }
    }
    schedulePoll(epoch, state.jobs.some((job) => ACTIVE_STATUSES.has(job.status)) ? 2000 : 10000);
  } catch (error) {
    handleConnectionError(error, epoch);
  } finally {
    if (epoch === generation) {
      $("connect-button").textContent = "Kết nối ↗";
      renderControls();
    }
  }
}

async function refreshJobs(epoch = generation) {
  if (activeRefresh?.epoch === epoch) return activeRefresh.promise;
  const promise = (async () => {
    const jobs = normalizeJobs(await api("/api/jobs"));
    if (epoch !== generation) return;
    const selectedId = nextSelectedId(jobs, state.selectedId);
    const selected = jobs.find((job) => job.id === selectedId);
    if (selected && ACTIVE_STATUSES.has(selected.status)) {
      const detail = normalizeJob((await api(`/api/jobs/${encodeURIComponent(selected.id)}`)).job);
      if (epoch !== generation) return;
      if (detail.id !== selected.id) throw new ApiError("Mã tác vụ trong phản hồi backend không khớp.");
      jobs[jobs.findIndex((job) => job.id === detail.id)] = detail;
    }
    state.jobs = jobs;
    state.selectedId = selectedId;
    if (state.uncertain) {
      state.uncertain = false;
      notice("Đã tải lại lịch sử. Kiểm tra tác vụ vừa gửi trước khi bắt đầu một lần phân tích mới.");
    }
    renderJobs();
    renderControls();
  })();
  activeRefresh = { epoch, promise };
  try { await promise; }
  finally { if (activeRefresh?.promise === promise) activeRefresh = null; }
}

function formatTime(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Chưa có thời gian" : new Intl.DateTimeFormat("vi-VN", {
    day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit",
  }).format(date);
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderJobs() {
  const focusedId = document.activeElement?.closest(".job-row")?.dataset.jobId;
  $("history-count").textContent = String(state.jobs.length);
  $("history-empty").hidden = state.jobs.length > 0;
  $("history-empty").textContent = state.connected
    ? "Chưa có tác vụ. Chọn một tài liệu để bắt đầu phân tích."
    : "Chưa có tác vụ. Kết nối backend và chọn tài liệu để bắt đầu.";
  const fragment = document.createDocumentFragment();
  for (const job of state.jobs) {
    const row = element("button", `job-row${job.id === state.selectedId ? " selected" : ""}`);
    row.type = "button";
    row.dataset.jobId = job.id;
    row.setAttribute("aria-pressed", String(job.id === state.selectedId));
    row.setAttribute("aria-label", `${job.filename}, ${STATUS_LABELS[job.status]}`);
    const extension = job.filename.split(".").pop()?.toUpperCase().slice(0, 4) ?? "FILE";
    const body = element("span", "job-row-body");
    body.append(element("span", "job-row-name", job.filename),
      element("span", "job-row-meta", `${formatTime(job.created_at)} · ${job.mode === "full" ? "Đầy đủ" : "Chạy nhanh · Không model học"}`));
    row.append(element("span", "job-row-icon", extension), body,
      element("span", `badge ${job.status}`, STATUS_LABELS[job.status]), element("span", "job-row-arrow", "↗"));
    row.addEventListener("click", () => {
      state.selectedId = job.id;
      renderJobs();
      if (state.connected && ACTIVE_STATUSES.has(job.status)) {
        refreshJobs().catch((error) => handleConnectionError(error, generation));
      }
    });
    fragment.append(row);
  }
  $("job-list").replaceChildren(fragment);
  if (focusedId) {
    const focusedRow = [...$("job-list").children].find((row) => row.dataset.jobId === focusedId);
    focusedRow?.focus({ preventScroll: true });
  }
  renderDetail();
}

function renderDetail() {
  const job = state.jobs.find((item) => item.id === state.selectedId);
  $("empty-detail").hidden = Boolean(job);
  $("job-detail").hidden = !job;
  resultLinks = {};
  if (!job) return;
  $("detail-status").className = `badge ${job.status}`;
  $("detail-status").textContent = STATUS_LABELS[job.status];
  $("detail-mode").textContent = MODE_LABELS[job.mode];
  $("detail-filename").textContent = job.filename;
  $("detail-time").textContent = `Bắt đầu ${formatTime(job.created_at)} · ${job.device.toUpperCase()}`;
  $("detail-stage").textContent = STAGE_LABELS[job.stage] || job.stage || {
    queued: "Tác vụ đã tiếp nhận, đang chờ bộ xử lý.",
    running: "Đang xử lý tài liệu. Bạn có thể để cửa sổ này mở để theo dõi.",
    completed: "Phân tích hoàn tất. Mở kết quả để đối chiếu nội dung và nguồn.",
    failed: "Tác vụ chưa hoàn tất. Xem thông báo và nhật ký bên dưới.",
  }[job.status];
  $("progress-track").className = `progress-track ${job.status}`;
  $("detail-error").hidden = !job.error;
  $("detail-error").textContent = job.error;
  $("detail-log").textContent = job.log_tail || "Chưa có nhật ký xử lý.";
  $("detail-id").textContent = job.id;
  for (const key of ["report", "workspace", "analysis"]) {
    const url = resultUrl(state.settings.baseUrl, job.results[key], job.id);
    if (url) resultLinks[key] = url;
  }
  $("result-actions").hidden = job.status !== "completed";
  $("open-report").hidden = !resultLinks.report;
  $("open-workspace").hidden = !resultLinks.workspace;
  $("download-analysis").hidden = !resultLinks.analysis;
  if (resultLinks.analysis) $("download-analysis").href = resultLinks.analysis;
  else $("download-analysis").removeAttribute("href");
  if (job.status === "completed" && Object.keys(resultLinks).length === 0) {
    $("detail-error").textContent = "Backend chưa trả về đường dẫn kết quả hợp lệ. Kiểm tra nhật ký hoặc làm mới lịch sử.";
    $("detail-error").hidden = false;
  }
}

function clearFile() {
  state.file = null;
  $("document-file").value = "";
  $("file-title").textContent = "Kéo thả tài liệu vào đây";
  $("file-description").replaceChildren(document.createTextNode("hoặc "), element("span", "file-link", "chọn tệp từ máy"));
  $("file-status").textContent = "Chưa chọn tài liệu";
  $("dropzone").classList.remove("selected");
  $("remove-file").hidden = true;
  renderControls();
}

function selectFile(file) {
  if (state.submitting) return;
  try {
    validateFile(file, state.maxUploadBytes);
    state.file = file;
    $("file-title").textContent = file.name;
    $("file-description").textContent = `${formatBytes(file.size)} · Nhấn để đổi tài liệu`;
    $("file-status").textContent = "Tài liệu đã sẵn sàng";
    $("dropzone").classList.add("selected");
    $("remove-file").hidden = false;
    renderControls();
  } catch (error) {
    clearFile();
    notice(error.message, true);
  }
}

function fileBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const value = String(reader.result ?? "");
      const comma = value.indexOf(",");
      if (comma === -1) reject(new Error("Không đọc được nội dung tài liệu."));
      else resolve(value.slice(comma + 1));
    };
    reader.onerror = () => reject(new Error("Không đọc được tài liệu. Hãy chọn lại tệp từ máy."));
    reader.onabort = () => reject(new Error("Việc đọc tài liệu đã bị hủy."));
    reader.readAsDataURL(file);
  });
}

async function submitAnalysis(event) {
  event.preventDefault();
  if (state.submitting || state.uncertain) return;
  if (!state.connected) { notice("Kết nối bộ xử lý trước khi gửi tài liệu.", true); return; }
  let sent = false;
  const epoch = generation;
  try {
    validateFile(state.file, state.maxUploadBytes);
    state.submitting = true;
    clearTimeout(pollTimer);
    renderControls();
    $("notice").hidden = true;
    const file = state.file;
    const content = await fileBase64(file);
    const payload = {
      filename: file.name, content_base64: content,
      mode: new FormData($("analysis-form")).get("mode"),
      device: $("device").value, ocr: $("ocr").value,
    };
    sent = true;
    const job = normalizeJob((await api("/api/jobs", { method: "POST", body: payload, timeout: 45000 })).job);
    if (epoch !== generation) return;
    state.jobs = [job, ...state.jobs.filter((item) => item.id !== job.id)];
    state.selectedId = job.id;
    clearFile();
    renderJobs();
    notice("Đã tiếp nhận tài liệu. Bạn có thể theo dõi tiến độ tại Không gian kết quả.");
    schedulePoll(epoch, 1000);
  } catch (error) {
    if (epoch !== generation) return;
    if (sent && (!error.status || error.status >= 500)) {
      state.uncertain = true;
      notice("Chưa xác nhận được backend đã nhận tài liệu. Ứng dụng sẽ kiểm tra lịch sử; không tự gửi lại để tránh tác vụ trùng.", true);
      handleConnectionError(error, epoch);
    } else {
      notice(error.message, true);
      if (error.status === 401 || error.status === 403) handleConnectionError(error, epoch);
      else schedulePoll(epoch, 2000);
    }
  } finally {
    state.submitting = false;
    renderControls();
  }
}

async function openResult(key) {
  const url = resultLinks[key];
  if (!url) return;
  try { await chrome.tabs.create({ url }); }
  catch { notice("Không mở được kết quả. Kiểm tra extension đã được nạp trong Chrome hoặc Edge.", true); }
}

$("connection-form").addEventListener("submit", (event) => { event.preventDefault(); connect(); });
$("connection-toggle").addEventListener("click", () => toggleConnection($("connection-form").hidden));
$("dismiss-notice").addEventListener("click", () => { $("notice").hidden = true; });
$("document-file").addEventListener("change", (event) => { if (event.target.files[0]) selectFile(event.target.files[0]); });
$("remove-file").addEventListener("click", clearFile);
$("analysis-form").addEventListener("submit", submitAnalysis);
$("open-report").addEventListener("click", () => openResult("report"));
$("open-workspace").addEventListener("click", () => openResult("workspace"));
$("refresh-jobs").addEventListener("click", () => {
  refreshJobs().catch((error) => handleConnectionError(error, generation));
});
$("dropzone").addEventListener("dragover", (event) => {
  event.preventDefault();
  if (!state.submitting) $("dropzone").classList.add("dragover");
});
$("dropzone").addEventListener("dragleave", () => $("dropzone").classList.remove("dragover"));
$("dropzone").addEventListener("drop", (event) => {
  event.preventDefault();
  $("dropzone").classList.remove("dragover");
  if (event.dataTransfer.files.length !== 1) { notice("Chọn một tài liệu cho mỗi tác vụ.", true); return; }
  selectFile(event.dataTransfer.files[0]);
});

async function initialize() {
  try {
    if (!globalThis.chrome?.storage?.local) {
      notice("Mở ứng dụng từ extension PSAI đã nạp trong Chrome hoặc Edge để lưu cấu hình và xem kết quả.");
      return;
    }
    const stored = (await chrome.storage.local.get("psaiSettings")).psaiSettings;
    if (stored) {
      state.settings = { baseUrl: validateBackendUrl(stored.baseUrl), token: validateToken(stored.token) };
      $("backend-url").value = state.settings.baseUrl;
      $("pairing-token").value = state.settings.token;
      await connect(false);
    }
  } catch {
    notice("Cấu hình đã lưu chưa hợp lệ. Nhập lại địa chỉ backend và mã ghép nối để tiếp tục.");
    toggleConnection(true);
  }
}

initialize();
