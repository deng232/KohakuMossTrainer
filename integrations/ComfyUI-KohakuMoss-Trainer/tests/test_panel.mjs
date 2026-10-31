import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

test("a late log response cannot replace the newly selected job's logs or cursor", async () => {
    const elements = [], timers = [], requests = [];
    let extension, releaseLogs;
    class Element {
        constructor(tag) {
            this.tag = tag;
            this.style = {};
            this.attributes = {};
            this.textContent = "";
            this.isConnected = true;
            elements.push(this);
        }
        append() {}
        replaceChildren() {}
        setAttribute(name, value) { this.attributes[name] = value; }
        getContext() { return { clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {} }; }
    }
    const jobs = ["a", "b"].map(job_id => ({
        job_id, run_name: job_id, status: "completed", settings: { steps: 2 },
        metrics: {}, exports: [], checkpoints: [], warnings: [],
    }));
    const response = data => ({ ok: true, json: async () => data });
    const app = { registerExtension(value) { extension = value; }, extensionManager: {} };
    const api = { async fetchApi(url) {
        requests.push(url);
        if (url === "/kohakumoss/jobs") return response({ jobs });
        if (url.includes("/a/logs")) return new Promise(resolve => {
            releaseLogs = () => resolve(response({ text: "old job log", offset: 77, generation: 0 }));
        });
        if (url.includes("/b/logs")) return response({ text: "new job log", offset: 11, generation: 0 });
        return response({ rows: [] });
    } };
    const source = readFileSync(new URL("../web/trainer.js", import.meta.url), "utf8")
        .replace(/^import .*;$/gm, "");
    vm.runInNewContext(source, {
        app, api, document: { createElement: tag => new Element(tag) }, AbortController,
        setTimeout(fn) { timers.push(fn); return timers.length; }, clearTimeout() {},
    });
    app.extensionManager.registerSidebarTab = tab => tab.render(new Element("host"));
    extension.setup();
    await new Promise(setImmediate);
    assert.equal(typeof releaseLogs, "function");
    const select = elements.find(node => node.tag === "select");
    const logs = elements.find(node => node.attributes["aria-label"] === "Recent worker logs");
    select.value = "b";
    select.onchange();
    releaseLogs();
    await new Promise(setImmediate);
    assert.equal(logs.textContent, "");
    await timers.shift()();
    assert.equal(logs.textContent, "new job log");
    assert.ok(requests.some(url => url.includes("/b/logs?offset=0&")));
});

test("new mode hides the checkpoint widget and restored resume mode shows it", () => {
    let extension;
    const source = readFileSync(new URL("../web/trainer.js", import.meta.url), "utf8").replace(/^import .*;$/gm, "");
    vm.runInNewContext(source, { app: { registerExtension(value) { extension = value; } } });
    const computeSize = () => [200, 34];
    const mode = { name: "start_mode", value: "new" };
    const checkpoint = { name: "resume_checkpoint", value: "saved/path", element: {}, computeSize };
    const node = { comfyClass: "KohakuMossTrainer", widgets: [mode, checkpoint], setDirtyCanvas() {} };
    extension.nodeCreated(node);
    assert.equal(checkpoint.hidden, true);
    assert.equal(checkpoint.element.hidden, true);
    assert.equal(checkpoint.computeSize()[1], -4);
    mode.value = "resume_latest";
    mode.callback();
    assert.equal(checkpoint.hidden, false);
    assert.equal(checkpoint.computeSize, computeSize);
    mode.value = "new";
    node.onConfigure();
    assert.equal(checkpoint.hidden, true);
    assert.equal(checkpoint.value, "saved/path");
});

test("learning rate stays editable below the frontend's default rounding precision", () => {
    let extension;
    const source = readFileSync(new URL("../web/trainer.js", import.meta.url), "utf8").replace(/^import .*;$/gm, "");
    vm.runInNewContext(source, { app: { registerExtension(value) { extension = value; } } });
    const learningRate = { name: "learning_rate", value: 0.0001, options: { precision: 1, round: 0.1 } };
    extension.nodeCreated({ comfyClass: "KohakuMossTrainer", widgets: [learningRate] });
    assert.equal(learningRate.options.round, false);
    assert.equal(learningRate.options.precision, 12);
    for (const value of [0.0001, 0.00005, 1e-12]) {
        assert.equal(Number(value.toFixed(learningRate.options.precision)), value);
    }
    assert.equal(learningRate.value, 0.0001);
});
