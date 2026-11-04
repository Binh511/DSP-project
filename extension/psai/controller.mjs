import { PSAIApiClient } from "./api-client.mjs";
import {
  ACTIVE_STATUSES,
  DEFAULT_BACKEND,
  MAX_UPLOAD_BYTES,
  nextSelectedId,
  resultUrl,
  validateBackendUrl,
  validateFile,
  validateToken,
} from "./core.mjs";

const ACTIVE_POLL_MS = 2_000;
const IDLE_POLL_MS = 10_000;
const RECONNECT_MS = 5_000;

/** Coordinates state and background requests; the view owns every DOM operation. */
export class PSAIController {
  constructor(view, { storage, openTab }) {
    this.view = view;
    this.storage = storage;
    this.openTab = openTab;
    this.state = {
      settings: { baseUrl: DEFAULT_BACKEND, token: "" },
      connected: false,
      connecting: false,
      jobs: [],
      selectedId: null,
      file: null,
      submitting: false,
      uncertain: false,
      maxUploadBytes: MAX_UPLOAD_BYTES,
    };
    this.api = null;
    this.generation = 0;
    this.jobsRevision = 0;
    this.pollTimer = null;
    this.activeRefresh = null;
    view.bind({
      connect: () => this.connect(),
      submit: () => this.submitAnalysis(),
      selectFile: (file) => this.selectFile(file),
      clearFile: () => this.clearFile(),
      refresh: () => this.refresh(),
      selectJob: (id) => this.selectJob(id),
      openResult: (key) => this.openResult(key),
    });
  }

  async initialize() {
    try {
      if (!this.storage) {
        this.view.notice("Mở ứng dụng từ extension PSAI đã nạp trong Chrome hoặc Edge để lưu cấu hình và xem kết quả.");
        return;
      }
      const stored = (await this.storage.get("psaiSettings")).psaiSettings;
      if (stored) {
        this.state.settings = this.validateSettings(stored);
        this.view.setSettings(this.state.settings);
        await this.connect(false);
      }
    } catch {
      this.view.notice("Cấu hình đã lưu chưa hợp lệ. Nhập lại địa chỉ backend và mã ghép nối để tiếp tục.");
      this.view.toggleConnection(true);
    }
  }

  validateSettings(settings) {
    return {
      baseUrl: validateBackendUrl(settings.baseUrl),
      token: validateToken(settings.token),
    };
  }

  setConnectionStatus(phase, message) {
    this.state.connected = phase === "online";
    this.state.connecting = phase === "connecting";
    this.view.renderConnection(phase, message);
    this.view.renderControls(this.state);
  }

  async persistSettings() {
    try {
      await this.storage.set({ psaiSettings: this.state.settings });
    } catch {
      this.view.notice("Không lưu được cài đặt vào trình duyệt. Bạn có thể cần nhập lại mã khi mở ứng dụng.");
    }
  }

  async connect(save = true) {
    let settings;
    try {
      settings = save ? this.validateSettings(this.view.readSettings()) : this.state.settings;
    } catch (error) {
      this.view.notice(error.message, true);
      return;
    }
    const epoch = ++this.generation;
    clearTimeout(this.pollTimer);
    const previous = this.state.settings;
    if (settings.baseUrl !== previous.baseUrl || settings.token !== previous.token) {
      this.state.jobs = [];
      this.state.selectedId = null;
      this.state.uncertain = false;
      this.view.renderJobs(this.state);
    }
    this.state.settings = settings;
    this.api = new PSAIApiClient(settings);
    const api = this.api;
    this.view.setSettings(settings);
    this.setConnectionStatus("connecting", "Đang kiểm tra kết nối với backend PSAI…");
    if (save) await this.persistSettings();
    try {
      const health = await api.health();
      if (epoch !== this.generation) return;
      this.state.maxUploadBytes = health.maxUploadBytes;
      this.view.renderUploadLimit(health.maxUploadBytes);
      await this.refreshJobs(epoch);
      if (epoch !== this.generation) return;
      this.setConnectionStatus("online", `Backend PSAI ${health.version} đang hoạt động tại ${settings.baseUrl}.`);
      this.view.renderJobs(this.state);
      this.view.toggleConnection(false);
      if (this.state.file) this.selectFile(this.state.file);
      this.schedulePoll(epoch, this.pollDelay());
    } catch (error) {
      this.handleConnectionError(error, epoch);
    } finally {
      if (epoch === this.generation) {
        this.state.connecting = false;
        this.view.renderControls(this.state);
      }
    }
  }

  pollDelay() {
    return this.state.jobs.some((job) => ACTIVE_STATUSES.has(job.status)) ? ACTIVE_POLL_MS : IDLE_POLL_MS;
  }

  schedulePoll(epoch, delay) {
    clearTimeout(this.pollTimer);
    this.pollTimer = setTimeout(async () => {
      if (epoch !== this.generation || this.state.submitting) return;
      if (!this.state.connected) {
        await this.connect(false);
        return;
      }
      const revision = this.jobsRevision;
      try {
        await this.refreshJobs(epoch);
        if (this.isCurrentRefresh(epoch, revision) && !this.state.submitting) {
          this.schedulePoll(epoch, this.pollDelay());
        }
      } catch (error) {
        if (this.isCurrentRefresh(epoch, revision)) this.handleConnectionError(error, epoch);
      }
    }, delay);
  }

  handleConnectionError(error, epoch) {
    if (epoch !== this.generation) return;
    if (error.status === 401 || error.status === 403) {
      this.setConnectionStatus("auth", error.message);
      this.view.toggleConnection(true);
      clearTimeout(this.pollTimer);
      return;
    }
    this.setConnectionStatus("error", `${error.message} Tự thử lại sau 5 giây.`);
    this.schedulePoll(epoch, RECONNECT_MS);
  }

  async refresh() {
    const epoch = this.generation;
    const revision = this.jobsRevision;
    try {
      await this.refreshJobs(epoch);
    } catch (error) {
      if (this.isCurrentRefresh(epoch, revision)) this.handleConnectionError(error, epoch);
    }
  }

  async refreshJobs(epoch = this.generation) {
    const revision = this.jobsRevision;
    if (this.activeRefresh?.epoch === epoch && this.activeRefresh.revision === revision) {
      return this.activeRefresh.promise;
    }
    const promise = this.loadJobs(epoch, revision).catch((error) => {
      // A superseded GET must not disconnect a backend that just accepted a job.
      if (this.isCurrentRefresh(epoch, revision)) throw error;
    });
    this.activeRefresh = { epoch, revision, promise };
    try {
      await promise;
    } finally {
      if (this.activeRefresh?.promise === promise) this.activeRefresh = null;
    }
  }

  isCurrentRefresh(epoch, revision) {
    return epoch === this.generation && revision === this.jobsRevision;
  }

  invalidateJobRefreshes() {
    this.jobsRevision++;
    // Detach the old promise so the next poll can start a fresh request at once.
    this.activeRefresh = null;
  }

  async loadJobs(epoch, revision) {
    const api = this.api;
    const jobs = await api.listJobs();
    if (!this.isCurrentRefresh(epoch, revision)) return;
    const selectedId = nextSelectedId(jobs, this.state.selectedId);
    const selected = jobs.find((job) => job.id === selectedId);
    if (selected && ACTIVE_STATUSES.has(selected.status)) {
      const detail = await api.getJob(selected.id);
      if (!this.isCurrentRefresh(epoch, revision)) return;
      jobs[jobs.findIndex((job) => job.id === detail.id)] = detail;
    }
    this.state.jobs = jobs;
    // The user may have selected another history row while detail was loading.
    this.state.selectedId = nextSelectedId(jobs, this.state.selectedId);
    if (this.state.uncertain) {
      this.state.uncertain = false;
      this.view.notice("Đã tải lại lịch sử. Kiểm tra tác vụ vừa gửi trước khi bắt đầu một lần phân tích mới.");
    }
    this.view.renderJobs(this.state);
    this.view.renderControls(this.state);
  }

  selectJob(id) {
    this.state.selectedId = id;
    this.view.renderJobs(this.state);
    const job = this.state.jobs.find((item) => item.id === id);
    if (this.state.connected && ACTIVE_STATUSES.has(job?.status)) this.refresh();
  }

  clearFile() {
    this.state.file = null;
    this.view.renderFile(null);
    this.view.renderControls(this.state);
  }

  selectFile(file) {
    if (this.state.submitting) return;
    try {
      this.state.file = validateFile(file, this.state.maxUploadBytes);
      this.view.renderFile(file);
      this.view.renderControls(this.state);
    } catch (error) {
      this.clearFile();
      this.view.notice(error.message, true);
    }
  }

  async submitAnalysis() {
    if (this.state.submitting || this.state.uncertain) return;
    if (!this.state.connected) {
      this.view.notice("Kết nối bộ xử lý trước khi gửi tài liệu.", true);
      return;
    }
    let sent = false;
    const epoch = this.generation;
    try {
      const file = validateFile(this.state.file, this.state.maxUploadBytes);
      const options = this.view.readAnalysisOptions();
      this.state.submitting = true;
      clearTimeout(this.pollTimer);
      this.view.renderControls(this.state);
      this.view.hideNotice();
      const content = await this.view.readFile(file);
      this.invalidateJobRefreshes();
      sent = true;
      const job = await this.api.createJob({
        filename: file.name,
        content_base64: content,
        ...options,
      });
      if (epoch !== this.generation) return;
      this.invalidateJobRefreshes();
      this.state.jobs = [job, ...this.state.jobs.filter((item) => item.id !== job.id)];
      this.state.selectedId = job.id;
      this.clearFile();
      this.view.renderJobs(this.state);
      this.view.notice("Đã tiếp nhận tài liệu. Bạn có thể theo dõi tiến độ tại Không gian kết quả.");
      this.schedulePoll(epoch, 1_000);
    } catch (error) {
      if (epoch !== this.generation) return;
      if (sent) this.invalidateJobRefreshes();
      if (sent && (!error.status || error.status >= 500)) {
        this.state.uncertain = true;
        this.view.notice("Chưa xác nhận được backend đã nhận tài liệu. Ứng dụng sẽ kiểm tra lịch sử; không tự gửi lại để tránh tác vụ trùng.", true);
        this.handleConnectionError(error, epoch);
      } else {
        this.view.notice(error.message, true);
        if (error.status === 401 || error.status === 403) this.handleConnectionError(error, epoch);
        else this.schedulePoll(epoch, ACTIVE_POLL_MS);
      }
    } finally {
      this.state.submitting = false;
      this.view.renderControls(this.state);
    }
  }

  async openResult(key) {
    const job = this.state.jobs.find((item) => item.id === this.state.selectedId);
    if (job?.status !== "completed") return;
    const url = resultUrl(this.state.settings.baseUrl, job.results[key], job.id);
    if (!url) return;
    try {
      await this.openTab(url);
    } catch {
      this.view.notice("Không mở được kết quả. Kiểm tra extension đã được nạp trong Chrome hoặc Edge.", true);
    }
  }
}
