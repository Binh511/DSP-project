import { formatBytes, resultUrl } from "./core.mjs";

const STATUS_LABELS = {
  queued: "Đang chờ",
  running: "Đang xử lý",
  completed: "Hoàn tất",
  failed: "Có lỗi",
};
const MODE_LABELS = {
  full: "Đầy đủ · AI cục bộ",
  fast: "Chạy nhanh · Không model học",
};
const CONNECTION_LABELS = {
  offline: "Chưa kết nối",
  connecting: "Đang kết nối",
  online: "Đã kết nối",
  error: "Mất kết nối",
  auth: "Kiểm tra mã",
};
const STAGE_LABELS = {
  queued: "Tài liệu đã tiếp nhận, đang chờ bộ xử lý.",
  source: "Đang đọc tài liệu và chuẩn bị văn bản nguồn…",
  extract: "Đang phân loại và trích xuất thông tin từ tài liệu…",
  retrieve: "Đang truy xuất, đối chiếu các đoạn nguồn liên quan…",
  plan: "Đang tổng hợp báo cáo và lập kế hoạch dự án…",
  completed: "Phân tích hoàn tất. Mở kết quả để đối chiếu nội dung và nguồn.",
  failed: "Tác vụ chưa hoàn tất. Xem thông báo và nhật ký bên dưới.",
};
const STATUS_DESCRIPTIONS = {
  queued: "Tác vụ đã tiếp nhận, đang chờ bộ xử lý.",
  running: "Đang xử lý tài liệu. Bạn có thể để cửa sổ này mở để theo dõi.",
  completed: STAGE_LABELS.completed,
  failed: STAGE_LABELS.failed,
};
const dateFormatter = new Intl.DateTimeFormat("vi-VN", {
  day: "2-digit",
  month: "2-digit",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

function formatTime(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Chưa có thời gian" : dateFormatter.format(date);
}

/** DOM rendering and browser file input. All backend/user strings use textContent. */
export class PSAIView {
  constructor(document) {
    this.document = document;
    this.onSelectJob = () => {};
  }

  element(id) {
    return this.document.getElementById(id);
  }

  createElement(tag, className, text) {
    const node = this.document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  bind(actions) {
    this.onSelectJob = actions.selectJob;
    this.element("connection-form").addEventListener("submit", (event) => {
      event.preventDefault();
      actions.connect();
    });
    this.element("analysis-form").addEventListener("submit", (event) => {
      event.preventDefault();
      actions.submit();
    });
    this.element("connection-toggle").addEventListener("click", () => {
      this.toggleConnection(this.element("connection-form").hidden);
    });
    this.element("dismiss-notice").addEventListener("click", () => this.hideNotice());
    this.element("document-file").addEventListener("change", (event) => {
      if (event.target.files[0]) actions.selectFile(event.target.files[0]);
    });
    this.element("remove-file").addEventListener("click", actions.clearFile);
    this.element("refresh-jobs").addEventListener("click", actions.refresh);
    this.element("open-report").addEventListener("click", () => actions.openResult("report"));
    this.element("open-workspace").addEventListener("click", () => actions.openResult("workspace"));
    this.bindDropzone(actions);
  }

  bindDropzone(actions) {
    const dropzone = this.element("dropzone");
    dropzone.addEventListener("dragover", (event) => {
      event.preventDefault();
      if (!this.element("document-file").disabled) dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
      if (event.dataTransfer.files.length !== 1) {
        this.notice("Chọn một tài liệu cho mỗi tác vụ.", true);
        return;
      }
      actions.selectFile(event.dataTransfer.files[0]);
    });
  }

  readSettings() {
    return {
      baseUrl: this.element("backend-url").value,
      token: this.element("pairing-token").value,
    };
  }

  setSettings(settings) {
    this.element("backend-url").value = settings.baseUrl;
    this.element("pairing-token").value = settings.token;
  }

  readAnalysisOptions() {
    return {
      mode: new FormData(this.element("analysis-form")).get("mode"),
      device: this.element("device").value,
      ocr: this.element("ocr").value,
    };
  }

  readFile(file) {
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

  notice(message, error = false) {
    this.element("notice-text").textContent = message;
    this.element("notice").classList.toggle("error", error);
    this.element("notice").hidden = false;
  }

  hideNotice() {
    this.element("notice").hidden = true;
  }

  toggleConnection(open) {
    this.element("connection-form").hidden = !open;
    this.element("connection-toggle").setAttribute("aria-expanded", String(open));
    this.element("connection-toggle").textContent = open ? "Thu gọn ↑" : "Cài đặt kết nối ↓";
  }

  renderConnection(phase, message) {
    const color = phase === "auth" ? "error" : phase;
    this.element("connection-dot").className = `status-dot ${color}`;
    this.element("connection-badge").className = `badge ${phase === "offline" ? "neutral" : color}`;
    this.element("connection-badge").textContent = CONNECTION_LABELS[phase];
    if (this.element("connection-message").textContent !== message) {
      this.element("connection-message").textContent = message;
    }
  }

  renderControls(state) {
    this.element("analyze-button").disabled = !state.connected || !state.file || state.submitting || state.uncertain;
    this.element("analyze-label").textContent = state.submitting ? "Đang gửi tài liệu…"
      : state.uncertain ? "Đang xác minh tác vụ…" : "Bắt đầu phân tích";
    this.element("refresh-jobs").disabled = !state.connected || state.submitting;
    this.element("connect-button").disabled = state.submitting || state.connecting;
    this.element("connect-button").textContent = state.connecting ? "Đang kết nối…" : "Kết nối ↗";
    for (const id of ["document-file", "remove-file", "backend-url", "pairing-token", "device", "ocr"]) {
      this.element(id).disabled = state.submitting;
    }
    for (const input of this.document.querySelectorAll('input[name="mode"]')) {
      input.disabled = state.submitting;
    }
  }

  renderUploadLimit(limit) {
    this.element("file-help").textContent = `PDF, Markdown, TXT · Tối đa ${formatBytes(limit)}`;
  }

  renderFile(file) {
    if (file) {
      this.element("file-title").textContent = file.name;
      this.element("file-description").textContent = `${formatBytes(file.size)} · Nhấn để đổi tài liệu`;
      this.element("file-status").textContent = "Tài liệu đã sẵn sàng";
    } else {
      this.element("document-file").value = "";
      this.element("file-title").textContent = "Kéo thả tài liệu vào đây";
      this.element("file-description").replaceChildren(
        this.document.createTextNode("hoặc "),
        this.createElement("span", "file-link", "chọn tệp từ máy"),
      );
      this.element("file-status").textContent = "Chưa chọn tài liệu";
    }
    this.element("dropzone").classList.toggle("selected", Boolean(file));
    this.element("remove-file").hidden = !file;
  }

  renderJobs(state) {
    const focusedId = this.document.activeElement?.closest(".job-row")?.dataset.jobId;
    this.element("history-count").textContent = String(state.jobs.length);
    this.element("history-empty").hidden = state.jobs.length > 0;
    this.element("history-empty").textContent = state.connected
      ? "Chưa có tác vụ. Chọn một tài liệu để bắt đầu phân tích."
      : "Chưa có tác vụ. Kết nối backend và chọn tài liệu để bắt đầu.";
    const fragment = this.document.createDocumentFragment();
    for (const job of state.jobs) fragment.append(this.createJobRow(job, state.selectedId));
    this.element("job-list").replaceChildren(fragment);
    if (focusedId) {
      const focusedRow = [...this.element("job-list").children].find((row) => row.dataset.jobId === focusedId);
      focusedRow?.focus({ preventScroll: true });
    }
    this.renderDetail(state);
  }

  createJobRow(job, selectedId) {
    const row = this.createElement("button", `job-row${job.id === selectedId ? " selected" : ""}`);
    row.type = "button";
    row.dataset.jobId = job.id;
    row.setAttribute("aria-pressed", String(job.id === selectedId));
    row.setAttribute("aria-label", `${job.filename}, ${STATUS_LABELS[job.status]}`);
    const extension = job.filename.split(".").pop()?.toUpperCase().slice(0, 4) ?? "FILE";
    const body = this.createElement("span", "job-row-body");
    body.append(
      this.createElement("span", "job-row-name", job.filename),
      this.createElement("span", "job-row-meta",
        `${formatTime(job.created_at)} · ${job.mode === "full" ? "Đầy đủ" : "Chạy nhanh · Không model học"}`),
    );
    row.append(
      this.createElement("span", "job-row-icon", extension),
      body,
      this.createElement("span", `badge ${job.status}`, STATUS_LABELS[job.status]),
      this.createElement("span", "job-row-arrow", "↗"),
    );
    row.addEventListener("click", () => this.onSelectJob(job.id));
    return row;
  }

  renderDetail(state) {
    const job = state.jobs.find((item) => item.id === state.selectedId);
    this.element("empty-detail").hidden = Boolean(job);
    this.element("job-detail").hidden = !job;
    if (!job) return;
    this.element("detail-status").className = `badge ${job.status}`;
    this.element("detail-status").textContent = STATUS_LABELS[job.status];
    this.element("detail-mode").textContent = MODE_LABELS[job.mode];
    this.element("detail-filename").textContent = job.filename;
    this.element("detail-time").textContent = `Bắt đầu ${formatTime(job.created_at)} · ${job.device.toUpperCase()}`;
    this.element("detail-stage").textContent = STAGE_LABELS[job.stage] || job.stage || STATUS_DESCRIPTIONS[job.status];
    this.element("progress-track").className = `progress-track ${job.status}`;
    this.element("detail-error").hidden = !job.error;
    this.element("detail-error").textContent = job.error;
    this.element("detail-log").textContent = job.log_tail || "Chưa có nhật ký xử lý.";
    this.element("detail-id").textContent = job.id;
    this.renderResults(job, state.settings.baseUrl);
  }

  renderResults(job, baseUrl) {
    const links = Object.fromEntries(["report", "workspace", "analysis"].map((key) => [
      key, resultUrl(baseUrl, job.results[key], job.id),
    ]));
    this.element("result-actions").hidden = job.status !== "completed";
    this.element("open-report").hidden = !links.report;
    this.element("open-workspace").hidden = !links.workspace;
    this.element("download-analysis").hidden = !links.analysis;
    if (links.analysis) this.element("download-analysis").href = links.analysis;
    else this.element("download-analysis").removeAttribute("href");
    if (job.status === "completed" && !Object.values(links).some(Boolean)) {
      this.element("detail-error").textContent = "Backend chưa trả về đường dẫn kết quả hợp lệ. Kiểm tra nhật ký hoặc làm mới lịch sử.";
      this.element("detail-error").hidden = false;
    }
  }
}
