import assert from "node:assert/strict";
import test from "node:test";
import { PSAIController } from "../controller.mjs";
import { ApiError } from "../core.mjs";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((accept, fail) => { resolve = accept; reject = fail; });
  return { promise, resolve, reject };
}

const job = (id, status = "completed") => ({
  id,
  filename: `${id}.md`,
  mode: "fast",
  device: "auto",
  ocr: "auto",
  status,
  stage: status,
  created_at: "2026-11-04T08:00:00Z",
  updated_at: "2026-11-04T08:00:00Z",
  error: "",
  log_tail: "",
  results: {},
});

function harness(t, jobs = [job("old")]) {
  const connections = [];
  const notices = [];
  const view = {
    bind() {},
    renderControls() {},
    renderJobs() {},
    renderFile() {},
    hideNotice() {},
    toggleConnection() {},
    readAnalysisOptions: () => ({ mode: "fast", device: "auto", ocr: "auto" }),
    readFile: async () => "eA==",
    renderConnection: (phase) => connections.push(phase),
    notice: (message) => notices.push(message),
  };
  const controller = new PSAIController(view, {
    storage: { get: async () => ({}), set: async () => {} },
    openTab: async () => {},
  });
  Object.assign(controller.state, {
    connected: true,
    file: { name: "proposal.md", size: 1 },
    jobs,
    selectedId: jobs[0]?.id ?? null,
  });
  t.after(() => clearTimeout(controller.pollTimer));
  return { controller, connections, notices };
}

test("a poll started before submission cannot remove the accepted job", async (t) => {
  const { controller } = harness(t);
  const history = deferred();
  const accepted = job("new", "queued");
  controller.api = {
    listJobs: () => history.promise,
    createJob: async () => accepted,
  };

  const refresh = controller.refreshJobs();
  await controller.submitAnalysis();
  assert.equal(controller.state.selectedId, accepted.id);
  history.resolve([job("old")]);
  await refresh;

  assert.deepEqual(controller.state.jobs.map((item) => item.id), ["new", "old"]);
  assert.equal(controller.state.selectedId, "new");
});

test("a fresh history request can finish before a superseded poll resolves", async (t) => {
  const { controller } = harness(t);
  const oldHistory = deferred();
  const accepted = job("new", "queued");
  let calls = 0;
  controller.api = {
    listJobs: () => ++calls === 1 ? oldHistory.promise : Promise.resolve([accepted, job("old")]),
    getJob: async () => job("new", "running"),
    createJob: async () => accepted,
  };

  const oldRefresh = controller.refreshJobs();
  await controller.submitAnalysis();
  await controller.refreshJobs();
  assert.equal(calls, 2);
  assert.equal(controller.state.jobs[0].status, "running");
  oldHistory.resolve([job("old")]);
  await oldRefresh;
  assert.equal(controller.state.jobs[0].id, "new");
  assert.equal(controller.state.jobs[0].status, "running");
});

test("a history selection made while another job detail loads is preserved", async (t) => {
  const jobs = [job("active", "queued"), job("chosen")];
  const { controller } = harness(t, jobs);
  const detail = deferred();
  const detailRequested = deferred();
  controller.api = {
    listJobs: async () => jobs,
    getJob: () => { detailRequested.resolve(); return detail.promise; },
  };

  const refresh = controller.refreshJobs();
  await detailRequested.promise;
  controller.selectJob("chosen");
  detail.resolve(job("active", "running"));
  await refresh;

  assert.equal(controller.state.selectedId, "chosen");
  assert.equal(controller.state.jobs[0].status, "running");
});

test("an error from an obsolete poll cannot disconnect a newly submitted job", async (t) => {
  const { controller, connections } = harness(t);
  const history = deferred();
  controller.api = {
    listJobs: () => history.promise,
    createJob: async () => job("new", "queued"),
  };

  const refresh = controller.refresh();
  await controller.submitAnalysis();
  history.reject(new ApiError("Old request failed", 503));
  await refresh;

  assert.equal(controller.state.connected, true);
  assert.equal(controller.state.selectedId, "new");
  assert.deepEqual(connections, []);
});

test("a pre-error snapshot cannot clear uncertainty after an ambiguous upload", async (t) => {
  const { controller } = harness(t);
  const upload = deferred();
  const uploadStarted = deferred();
  const history = deferred();
  controller.api = {
    listJobs: () => history.promise,
    createJob: () => { uploadStarted.resolve(); return upload.promise; },
  };

  const submission = controller.submitAnalysis();
  await uploadStarted.promise;
  const refresh = controller.refreshJobs();
  upload.reject(new ApiError("Upload connection lost"));
  await submission;
  assert.equal(controller.state.uncertain, true);
  history.resolve([job("old")]);
  await refresh;
  assert.equal(controller.state.uncertain, true);
});

test("an error from the current refresh still triggers authentication recovery", async (t) => {
  const { controller, connections } = harness(t);
  controller.api = { listJobs: async () => { throw new ApiError("Pair again", 401); } };

  await controller.refresh();

  assert.equal(controller.state.connected, false);
  assert.deepEqual(connections, ["auth"]);
  assert.equal(controller.pollTimer, null);
});
