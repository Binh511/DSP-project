import assert from "node:assert/strict";
import test from "node:test";
import {
  ApiError, MAX_UPLOAD_BYTES, nextSelectedId, normalizeHealth, normalizeJob,
  normalizeJobs, parseApiResponse, resultUrl, validateBackendUrl, validateFile, validateToken,
} from "../core.mjs";

const makeJob = (overrides = {}) => ({
  id: "a1b2", filename: "đề-xuất.md", mode: "full", device: "auto", ocr: "auto",
  status: "queued", stage: "Đang chờ", created_at: "2026-10-31T08:00:00Z",
  updated_at: "2026-10-31T08:00:00Z", error: null, log_tail: "", results: {}, ...overrides,
});

test("backend URL allows only a literal loopback address with a valid explicit port", () => {
  assert.equal(validateBackendUrl(" http://127.0.0.1:8765/ "), "http://127.0.0.1:8765");
  assert.equal(validateBackendUrl("http://127.0.0.1:65535"), "http://127.0.0.1:65535");
  for (const url of [
    "http://localhost:8765", "https://127.0.0.1:8765", "http://127.0.0.1", "http://127.0.0.1:0",
    "http://127.0.0.1:65536", "http://127.0.0.1:8765/api", "http://127.0.0.1:8765?x=1",
    "http://127.0.0.1:8765/#secret", "http://user@127.0.0.1:8765", "http://127.1:8765",
    "http://2130706433:8765", "http://127.0.0.1:8765\\evil", "http://[::1]:8765",
  ]) assert.throws(() => validateBackendUrl(url), undefined, url);
});

test("pairing token cannot contain whitespace, newlines or header-control characters", () => {
  assert.equal(validateToken("  abcdef0123456789  "), "abcdef0123456789");
  for (const token of ["", "short", "abcdef01 23456789", "abcdef0123456789\r\nx", "a".repeat(513)]) {
    assert.throws(() => validateToken(token));
  }
});

test("file validation enforces supported extensions, nonempty data and both upload limits", () => {
  assert.equal(validateFile({ name: "Đề xuất.PDF", size: MAX_UPLOAD_BYTES }).size, MAX_UPLOAD_BYTES);
  for (const file of [null, { name: "a.exe", size: 1 }, { name: "a.txt", size: 0 },
    { name: "a.pdf", size: MAX_UPLOAD_BYTES + 1 }, { name: "a.md", size: NaN }]) {
    assert.throws(() => validateFile(file));
  }
  assert.throws(() => validateFile({ name: "a.md", size: 1001 }, 1000));
});

test("health validation honors a smaller server upload limit", () => {
  assert.deepEqual(normalizeHealth({ status: "ok", version: "0.1.0", max_upload_bytes: 1024 }),
    { version: "0.1.0", maxUploadBytes: 1024 });
  assert.equal(normalizeHealth({ status: "ok", version: "0.1.0", max_upload_bytes: MAX_UPLOAD_BYTES * 2 }).maxUploadBytes, MAX_UPLOAD_BYTES);
  assert.throws(() => normalizeHealth({ status: "ok", version: "0.1.0", max_upload_bytes: -1 }), ApiError);
});

test("jobs are validated, sorted and unfinished jobs cannot expose results", () => {
  const jobs = normalizeJobs({ jobs: [makeJob(), makeJob({ id: "second", created_at: "2026-10-31T09:00:00Z" })] });
  assert.equal(jobs[0].id, "second");
  assert.deepEqual(normalizeJob(makeJob({ results: { report: "/untrusted" } })).results, {});
  assert.equal(normalizeJob(makeJob({ log_tail: ["first", "second"] })).log_tail, "first\nsecond");
  for (const override of [{ id: "../escape" }, { id: undefined }, { id: null }, { status: "unknown" }, { mode: "unknown" }, { ocr: "invalid" }, { device: "invalid" }]) {
    assert.throws(() => normalizeJob(makeJob(override)), ApiError);
  }
  assert.throws(() => normalizeJobs({ jobs: null }), ApiError);
});

test("selection keeps the current task and otherwise prefers an active task", () => {
  const completed = makeJob({ id: "complete", status: "completed" });
  const running = makeJob({ id: "running", status: "running" });
  assert.equal(nextSelectedId([completed, running], "complete"), "complete");
  assert.equal(nextSelectedId([completed, running], "gone"), "running");
  assert.equal(nextSelectedId([completed], null), "complete");
  assert.equal(nextSelectedId([], "gone"), null);
});

test("result links remain in the same backend and task capability path", () => {
  const base = "http://127.0.0.1:8765";
  assert.equal(resultUrl(base, "/results/a1b2/secret-token/report.html", "a1b2"), `${base}/results/a1b2/secret-token/report.html`);
  for (const path of [
    "https://evil.test/report.html", "//evil.test/results/a1b2/secret/report.html",
    "/results/other/secret/report.html", "/results/a1b2/../report.html",
    "/results/a1b2/secret/../../app.html", "/results/a1b2/secret/%2e%2e",
    "/results/a1b2/secret/report.html?token=leak", "/results/a1b2/secret/report.html#x",
    "/results/a1b2\\secret\\report.html", "/api/health", null,
  ]) assert.equal(resultUrl(base, path, "a1b2"), null, String(path));
});

test("API parser surfaces authentication, structured server errors and malformed responses", async () => {
  assert.deepEqual(await parseApiResponse(new Response('{"job":{"id":"x"}}', { status: 202 })), { job: { id: "x" } });
  await assert.rejects(parseApiResponse(new Response('{"error":"no"}', { status: 401 })), (error) => error.status === 401 && /Mã ghép nối/.test(error.message));
  await assert.rejects(parseApiResponse(new Response("forbidden", { status: 403 })), (error) => error.status === 403 && /Mã ghép nối/.test(error.message));
  await assert.rejects(parseApiResponse(new Response('{"error":{"message":"OCR missing"}}', { status: 422 })), /OCR missing/);
  await assert.rejects(parseApiResponse(new Response("not json", { status: 200 })), /không phải JSON/);
  await assert.rejects(parseApiResponse(new Response("[]", { status: 200 })), /không hợp lệ/);
});
