// Real unpacked-extension smoke test using Edge and Node's built-in CDP client.
// No browser libraries or npm installation are required (Node >= 22).
import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const edge = process.env.PSAI_EDGE_PATH || "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const output = path.join(root, "outputs/extension", `browser_${Date.now()}`);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const children = [];
let browser;

async function freePort() {
  const server = net.createServer();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  await new Promise((resolve) => server.close(resolve));
  return port;
}

async function until(check, label, timeout = 30000) {
  const started = Date.now();
  while (Date.now() - started < timeout) {
    const result = await check();
    if (result) return result;
    await sleep(150);
  }
  throw new Error(`Timed out: ${label}`);
}

function launch(executable, args) {
  const child = spawn(executable, args, { cwd: root, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
  child.on("error", (error) => { child.launchError = error; });
  children.push(child);
  return child;
}

class CDP {
  constructor(socket) {
    this.socket = socket;
    this.sequence = 0;
    this.pending = new Map();
    this.errors = [];
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(String(event.data));
      if (message.method === "Runtime.exceptionThrown") {
        const details = message.params.exceptionDetails;
        this.errors.push(details.exception?.description || details.text);
      }
      const pending = this.pending.get(message.id);
      if (!pending) return;
      clearTimeout(pending.timer);
      this.pending.delete(message.id);
      if (message.error) pending.reject(new Error(JSON.stringify(message.error)));
      else pending.resolve(message.result);
    });
  }
  static async connect(url) {
    const socket = new WebSocket(url);
    await new Promise((resolve, reject) => {
      socket.addEventListener("open", resolve, { once: true });
      socket.addEventListener("error", reject, { once: true });
    });
    return new CDP(socket);
  }
  call(method, params = {}) {
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 15000);
      this.pending.set(id, { resolve, reject, timer });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }
  async evaluate(expression) {
    const value = await this.call("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (value.exceptionDetails) throw new Error(value.exceptionDetails.exception?.description || value.exceptionDetails.text);
    return value.result.value;
  }
  close() { this.socket.close(); }
}

try {
  if (!existsSync(edge)) throw new Error("Set PSAI_EDGE_PATH to an installed Chromium/Edge executable.");
  await mkdir(output, { recursive: true });
  const backendPort = await freePort();
  const debugPort = await freePort();
  const python = path.join(root, ".venv/psai_runtime/Scripts/python.exe");
  const backend = launch(existsSync(python) ? python : "python", ["-u", "-m", "psai_extension",
    "--port", String(backendPort), "--data-dir", path.join(output, "jobs")]);
  let backendText = "";
  backend.stdout.on("data", (chunk) => { backendText += chunk.toString("utf8"); });
  const token = await until(() => {
    if (backend.launchError) throw backend.launchError;
    if (backend.exitCode !== null) throw new Error("Backend exited during browser setup.");
    return /Pairing token: ([A-Za-z0-9_-]+)/.exec(backendText)?.[1];
  }, "backend pairing token");
  const extensionPath = path.join(root, "extension/psai");
  const edgeProcess = launch(edge, ["--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
    `--user-data-dir=${path.join(output, "profile")}`, `--remote-debugging-port=${debugPort}`,
    `--disable-extensions-except=${extensionPath}`, `--load-extension=${extensionPath}`, "about:blank"]);
  const debugOrigin = `http://127.0.0.1:${debugPort}`;
  const version = await until(async () => {
    if (edgeProcess.launchError) throw edgeProcess.launchError;
    try { return await (await fetch(debugOrigin + "/json/version")).json(); }
    catch { return false; }
  }, "Edge debugging endpoint");
  browser = await CDP.connect(version.webSocketDebuggerUrl);
  const extension = await until(async () => {
    const { targetInfos } = await browser.call("Target.getTargets");
    return targetInfos.find((target) => target.type === "service_worker" && /^chrome-extension:\/\/[a-p]{32}\/service_worker\.js$/.test(target.url));
  }, "unpacked extension service worker");
  const extensionId = new URL(extension.url).hostname;
  console.log("Unpacked extension loaded in Edge:", extensionId);
  const { targetId } = await browser.call("Target.createTarget", { url: `chrome-extension://${extensionId}/app.html` });
  const pageInfo = await until(async () => (await (await fetch(debugOrigin + "/json/list")).json()).find((item) => item.id === targetId), "extension tab");
  const page = await CDP.connect(pageInfo.webSocketDebuggerUrl);
  await page.call("Runtime.enable");
  await page.call("Page.enable");
  await until(() => page.evaluate("document.readyState === 'complete' && Boolean(document.getElementById('connect-button'))"), "extension UI");
  await page.evaluate(`document.getElementById('backend-url').value=${JSON.stringify(`http://127.0.0.1:${backendPort}`)};
    document.getElementById('pairing-token').value=${JSON.stringify(token)};
    document.getElementById('connection-form').requestSubmit();`);
  await until(async () => {
    const connection = await page.evaluate("({badge:document.getElementById('connection-badge').textContent,message:document.getElementById('connection-message').textContent})");
    if (connection.badge === 'Kiểm tra mã' || connection.badge === 'Mất kết nối') throw new Error(connection.message);
    return connection.badge === 'Đã kết nối';
  }, "extension to backend connection");
  console.log("Extension connected to authenticated loopback backend.");
  const { root: documentRoot } = await page.call("DOM.getDocument");
  const { nodeId } = await page.call("DOM.querySelector", { nodeId: documentRoot.nodeId, selector: "#document-file" });
  await page.call("DOM.setFileInputFiles", { nodeId, files: [path.join(root, "examples/proposal.md")] });
  await page.evaluate("document.querySelector('input[name=mode][value=fast]').checked=true; document.getElementById('analysis-form').requestSubmit()");
  await until(() => page.evaluate("document.getElementById('detail-status').textContent === 'Hoàn tất'"), "actual pipeline result in extension", 90000);
  if (!await page.evaluate("!document.getElementById('open-report').hidden && !document.getElementById('open-workspace').hidden && !document.getElementById('download-analysis').hidden")) {
    throw new Error("Completed extension job is missing result actions.");
  }
  console.log("Upload, actual fast pipeline, polling and result actions passed.");
  await page.evaluate("document.getElementById('open-workspace').click()");
  const workspaceTarget = await until(async () => {
    const { targetInfos } = await browser.call("Target.getTargets");
    return targetInfos.find((target) => target.type === "page" && target.url.startsWith(`http://127.0.0.1:${backendPort}/results/`) && target.url.endsWith("workspace.html"));
  }, "workspace browser tab");
  const workspace = await CDP.connect(`ws://127.0.0.1:${debugPort}/devtools/page/${workspaceTarget.targetId}`);
  await workspace.call("Runtime.enable");
  await until(() => workspace.evaluate("document.readyState==='complete' && document.body.textContent.includes('Không gian dự án') && Boolean(document.getElementById('export'))"), "workspace rendered");
  const screenshot = await page.call("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  await writeFile(path.join(output, "extension-desktop.png"), Buffer.from(screenshot.data, "base64"));
  await page.call("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: false });
  const overflow = await page.evaluate("document.documentElement.scrollWidth > window.innerWidth");
  if (overflow) throw new Error("Extension UI overflows at 390px.");
  const mobile = await page.call("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  await writeFile(path.join(output, "extension-mobile.png"), Buffer.from(mobile.data, "base64"));
  await page.call("Page.reload");
  await until(() => page.evaluate("document.getElementById('connection-badge')?.textContent==='Đã kết nối' && document.getElementById('detail-status')?.textContent==='Hoàn tất'"), "saved settings and history after reload");
  if (page.errors.length || workspace.errors.length) throw new Error("Browser JavaScript errors: " + [...page.errors, ...workspace.errors].join("; "));
  await writeFile(path.join(output, "acceptance.json"), JSON.stringify({ status: "passed", browser: version.Browser,
    extension_id: extensionId, manifest_version: 3, mode: "fast", actual_unpacked_extension: true,
    authenticated_connection: true, file_upload: true, real_pipeline_executed: true, result_tab_opened: true,
    saved_settings_and_history_after_reload: true, responsive_width_390: true, javascript_errors: [] }, null, 2));
  console.log("Real Edge extension acceptance passed:", path.join(output, "acceptance.json"));
  page.close(); workspace.close();
  await browser.call("Browser.close");
  browser.close();
} finally {
  for (const child of children.reverse()) {
    if (child.exitCode !== null || !child.pid) continue;
    if (process.platform === "win32") spawnSync("taskkill.exe", ["/PID", String(child.pid), "/T", "/F"], { windowsHide: true, stdio: "ignore" });
    else child.kill("SIGTERM");
  }
}
