import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

test("choosing a full checkpoint selects its job and resume mode without uploading files", async () => {
    const elements = [];
    class Element {
        constructor(tag) { this.tag = tag; this.style = {}; this.textContent = ""; this.events = {}; this.value = ""; elements.push(this); }
        append() {}
        replaceChildren() {}
        setAttribute() {}
        addEventListener(name, callback) { this.events[name] = callback; }
        showModal() {}
        close() { this.events.close?.(); }
        remove() {}
    }
    const path = "/workspace/runs/run/checkpoints/step_00002000";
    const context = {
        api: { fetchApi: async url => {
            assert.equal(url, "/kohakumoss/jobs");
            return { ok: true, json: async () => ({ jobs: [{ job_id: "job1", run_name: "run", status: "completed", can_resume: true, checkpoints: [path] }] }) };
        } },
        document: { createElement: tag => new Element(tag), body: { append() {} } },
    };
    const source = readFileSync(new URL("../web/checkpoint_picker.js", import.meta.url), "utf8")
        .replace(/^import .*;$/gm, "").replace(/^export /gm, "");
    vm.runInNewContext(source, context);
    const widgets = [{ name: "start_mode", value: "new" }, { name: "resume_job_id", value: "" }];
    let changed = 0;
    const node = {
        widgets, graph: { change() { changed++; } }, setDirtyCanvas() {},
        addDOMWidget(name, type, root, options) {
            return { get value() { return options.getValue(); }, set value(value) { options.setValue(value); } };
        },
    };
    const { widget } = context.checkpointWidget(node, "resume_checkpoint", ["STRING", { default: "latest" }]);
    const choosing = elements.find(el => el.tag === "button").onclick();
    await new Promise(setImmediate);
    elements.find(el => el.tag === "button" && el.textContent.includes("step_00002000")).onclick();
    await choosing;
    assert.equal(widget.value, path);
    assert.equal(widgets[0].value, "resume_latest");
    assert.equal(widgets[1].value, "job1");
    assert.equal(changed, 1);
});
