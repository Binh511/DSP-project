import assert from "node:assert/strict";
import http from "node:http";
import test from "node:test";
import { PSAIApiClient } from "../api-client.mjs";
import { ApiError, MAX_UPLOAD_BYTES } from "../core.mjs";

const TOKEN = "test-pairing-token-0123456789";
const makeJob = (overrides = {}) => ({
  id: "job_123",
  filename: "đề-xuất.md",
  mode: "fast",
  device: "auto",
  ocr: "auto",
  status: "queued",
  stage: "queued",
  created_at: "2026-11-04T08:00:00Z",
  updated_at: "2026-11-04T08:00:00Z",
  error: null,
  log_tail: "",
  results: {},
  ...overrides,
});

async function serve(t, handler) {
  const server = http.createServer(handler);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(async () => {
    server.closeAllConnections();
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  });
  const settings = { baseUrl: `http://127.0.0.1:${server.address().port}`, token: TOKEN };
  return { client: new PSAIApiClient(settings), settings };
}

function json(response, body, status = 200) {
  response.writeHead(status, { "Content-Type": "application/json" });
  response.end(JSON.stringify(body));
}

test("real health request sends bearer auth in headers and normalizes the server limit", async (t) => {
  let observed;
  const { client } = await serve(t, (request, response) => {
    observed = { url: request.url, method: request.method, auth: request.headers.authorization };
    json(response, { status: "ok", version: "0.2.0", max_upload_bytes: 1024 });
  });
  assert.deepEqual(await client.health(), { version: "0.2.0", maxUploadBytes: 1024 });
  assert.deepEqual(observed, { url: "/api/health", method: "GET", auth: `Bearer ${TOKEN}` });
  assert.equal(observed.url.includes(TOKEN), false);
});

test("real upload posts JSON exactly once and normalizes accepted job data", async (t) => {
  let calls = 0;
  let received;
  const payload = {
    filename: "đề-xuất.md",
    content_base64: Buffer.from("# Đề xuất\nNội dung").toString("base64"),
    mode: "fast",
    device: "cpu",
    ocr: "none",
  };
  const { client } = await serve(t, (request, response) => {
    calls++;
    assert.equal(request.method, "POST");
    assert.equal(request.url, "/api/jobs");
    assert.equal(request.headers["content-type"], "application/json");
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk) => { body += chunk; });
    request.on("end", () => {
      received = JSON.parse(body);
      json(response, { job: makeJob({ device: "cpu", ocr: "none" }) }, 202);
    });
  });
  assert.equal((await client.createJob(payload)).id, "job_123");
  assert.deepEqual(received, payload);
  assert.equal(calls, 1);
});

test("job history sorts responses and detail rejects mismatched task identity", async (t) => {
  const { client } = await serve(t, (request, response) => {
    if (request.url === "/api/jobs") {
      json(response, { jobs: [
        makeJob(),
        makeJob({ id: "later", created_at: "2026-11-04T09:00:00Z" }),
      ] });
    } else {
      json(response, { job: makeJob({ id: "wrong_id" }) });
    }
  });
  assert.deepEqual((await client.listJobs()).map((job) => job.id), ["later", "job_123"]);
  await assert.rejects(client.getJob("job_123"), /không khớp/);
});

test("unsafe backend addresses and job paths never issue network requests", async (t) => {
  let calls = 0;
  const { settings } = await serve(t, (_request, response) => {
    calls++;
    json(response, { job: makeJob() });
  });
  for (const baseUrl of [
    "https://127.0.0.1:8765", "http://localhost:8765", "http://127.0.0.1:8765/api",
    "http://127.0.0.1:8765?token=leak", "http://user:pass@127.0.0.1:8765",
  ]) assert.throws(() => new PSAIApiClient({ ...settings, baseUrl }));
  const client = new PSAIApiClient(settings);
  for (const id of ["../health", "//other", "id?token=leak", "id#fragment", "id\\path", null]) {
    await assert.rejects(client.getJob(id), /Mã tác vụ không hợp lệ/);
  }
  assert.equal(calls, 0);
});

test("client captures validated settings instead of following later object mutations", async (t) => {
  const { client, settings } = await serve(t, (request, response) => {
    assert.equal(request.headers.authorization, `Bearer ${TOKEN}`);
    json(response, { status: "ok", version: "0.2.0", max_upload_bytes: MAX_UPLOAD_BYTES });
  });
  settings.baseUrl = "https://external.invalid";
  settings.token = "changed";
  assert.equal((await client.health()).version, "0.2.0");
});

test("HTTP auth errors survive non-JSON bodies and do not trigger automatic retries", async (t) => {
  let calls = 0;
  const { client } = await serve(t, (_request, response) => {
    calls++;
    response.writeHead(401, { "Content-Type": "text/plain" });
    response.end("Not authorized");
  });
  await assert.rejects(client.health(), (error) => error instanceof ApiError
    && error.status === 401 && /Mã ghép nối/.test(error.message));
  assert.equal(calls, 1);
});

test("structured HTTP errors and malformed success payloads remain explicit", async (t) => {
  let calls = 0;
  const { client } = await serve(t, (_request, response) => {
    calls++;
    if (calls === 1) json(response, { error: { message: "OCR missing" } }, 422);
    else json(response, { status: "ok", max_upload_bytes: -1 });
  });
  await assert.rejects(client.health(), (error) => error.status === 422 && error.message === "OCR missing");
  await assert.rejects(client.health(), /thông tin kết nối không hợp lệ/);
});

test("redirects are rejected before a second endpoint can receive the bearer token", async (t) => {
  const paths = [];
  const { client } = await serve(t, (request, response) => {
    paths.push(request.url);
    response.writeHead(302, { Location: "/other-endpoint" });
    response.end();
  });
  await assert.rejects(client.health(), (error) => error.status === 0 && /Không liên lạc được/.test(error.message));
  assert.deepEqual(paths, ["/api/health"]);
});

test("timeout aborts a real pending response without automatically resubmitting", async (t) => {
  let calls = 0;
  const { client } = await serve(t, () => { calls++; });
  await assert.rejects(client.createJob({ filename: "a.md" }, { timeout: 100 }), (error) =>
    error.status === 0 && /chưa phản hồi kịp/.test(error.message));
  assert.equal(calls, 1);
});

test("timeout also covers reading a response body after HTTP headers arrive", async (t) => {
  const { client } = await serve(t, (_request, response) => {
    response.writeHead(200, { "Content-Type": "application/json" });
    response.flushHeaders();
    response.write('{"status":');
  });
  await assert.rejects(client.health({ timeout: 100 }), (error) =>
    error.status === 0 && /chưa phản hồi kịp/.test(error.message));
});

test("transport failure is localized and exposes no pairing token", async () => {
  const client = new PSAIApiClient({ baseUrl: "http://127.0.0.1:8765", token: TOKEN }, {
    fetchImpl: async () => { throw new TypeError(`network failed for ${TOKEN}`); },
  });
  await assert.rejects(client.health(), (error) => error.status === 0
    && /Không liên lạc được/.test(error.message) && !error.message.includes(TOKEN));
});

test("fetch keeps its global receiver when stored on the API client", async () => {
  let calls = 0;
  const client = new PSAIApiClient({ baseUrl: "http://127.0.0.1:8765", token: TOKEN }, {
    fetchImpl: function () {
      assert.equal(this, globalThis, "Window.fetch requires its global receiver");
      calls++;
      return Promise.resolve(new Response(JSON.stringify({
        status: "ok", version: "0.2.0", max_upload_bytes: MAX_UPLOAD_BYTES,
      }), { headers: { "Content-Type": "application/json" } }));
    },
  });
  assert.equal((await client.health()).version, "0.2.0");
  assert.equal(calls, 1);
});
